#include <vulkan/vulkan.hpp>
#include <cstdlib>
#include <algorithm>
#include <array>
#include <cstring>
#include <cwchar>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <vector>

static vk::UniqueShaderModule shader(vk::Device dev, const std::filesystem::path & path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    const std::streamoff size = file.tellg();
    if (!file || size <= 0 || size % 4) throw std::runtime_error("invalid SPIR-V");
    std::vector<uint32_t> bytes(size/4);
    file.seekg(0);
    if (!file.read(reinterpret_cast<char *>(bytes.data()), size)) throw std::runtime_error("short SPIR-V");
    return dev.createShaderModuleUnique({{}, size_t(size), bytes.data()});
}

struct Buffer {
    vk::UniqueDeviceMemory memory;
    vk::UniqueBuffer buffer;
};

static Buffer make_buffer(vk::Device dev, const vk::PhysicalDeviceMemoryProperties & memory,
        vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties) {
    Buffer out;
    out.buffer = dev.createBufferUnique({{}, size, usage, vk::SharingMode::eExclusive});
    const auto requirements = dev.getBufferMemoryRequirements(*out.buffer);
    for (uint32_t type = 0; type < memory.memoryTypeCount; ++type) {
        if ((requirements.memoryTypeBits & (1u << type)) &&
            (memory.memoryTypes[type].propertyFlags & properties) == properties) {
            out.memory = dev.allocateMemoryUnique({requirements.size,type});
            dev.bindBufferMemory(*out.buffer,*out.memory,0);
            return out;
        }
    }
    throw std::runtime_error("no compatible memory type");
}


#include <limits>
#include <cmath>

struct Parameters {
    uint32_t blocks, steps, samples, bits, seed;
    float target;
};
static_assert(sizeof(Parameters)==24);

static uint32_t integer(const wchar_t * value, unsigned max) {
    wchar_t * end=nullptr;
    const auto result=std::wcstoull(value,&end,10);
    if (value==end || *end || result>max || *value==L'-') throw std::runtime_error("invalid integer parameter");
    return uint32_t(result);
}

static bool is_polaris(const vk::PhysicalDeviceProperties & properties) {
    return properties.vendorID==0x1002 && (properties.deviceID==0x67df || properties.deviceID==0x6fdf);
}

static void probe(vk::Instance instance) {
    for (const auto device : instance.enumeratePhysicalDevices()) {
        const auto props=device.getProperties();
        if (props.deviceType!=vk::PhysicalDeviceType::eDiscreteGpu &&
            props.deviceType!=vk::PhysicalDeviceType::eIntegratedGpu) continue;
        const auto memory=device.getMemoryProperties();
        const auto extensions=device.enumerateDeviceExtensionProperties();
        const bool reports_budget=props.apiVersion>=VK_API_VERSION_1_1 && std::any_of(extensions.begin(),extensions.end(),[](const auto & ext) {
            return std::strcmp(ext.extensionName.data(),VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)==0;
        });
        vk::PhysicalDeviceMemoryBudgetPropertiesEXT budget;
        if (reports_budget) {
            vk::PhysicalDeviceMemoryProperties2 properties;
            properties.pNext=&budget;
            device.getMemoryProperties2(&properties);
        }
        vk::DeviceSize capacity=0,available=0;
        for (uint32_t i=0;i<memory.memoryHeapCount;++i) {
            if (!(memory.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal)) continue;
            capacity+=memory.memoryHeaps[i].size;
            if (reports_budget && budget.heapBudget[i]>budget.heapUsage[i]) {
                available+=std::min(memory.memoryHeaps[i].size,budget.heapBudget[i]-budget.heapUsage[i]);
            }
        }
        const auto queues=device.getQueueFamilyProperties();
        const bool compute=std::any_of(queues.begin(),queues.end(),[](const auto & queue) {
            return queue.timestampValidBits && bool(queue.queueFlags & vk::QueueFlagBits::eCompute);
        });
        std::cout << props.vendorID << '\t' << props.deviceID << '\t'
                  << (props.deviceType==vk::PhysicalDeviceType::eDiscreteGpu ? capacity : 0) << '\t'
                  << (reports_budget && props.deviceType==vk::PhysicalDeviceType::eDiscreteGpu) << '\t' << available << '\t'
                  << (compute && is_polaris(props)) << '\t' << std::quoted(props.deviceName.data()) << '\n';
    }
}

int wmain(int argc, wchar_t ** argv) try {
    const vk::ApplicationInfo app("GGUF Converter GSQ and RCO",1,"",0,VK_API_VERSION_1_1);
    if (argc==2 && std::wcscmp(argv[1],L"--probe")==0) {
        const auto instance=vk::createInstanceUnique({{},&app});
        probe(*instance);
        return 0;
    }
    if (argc!=12) throw std::runtime_error("usage: vulkan-gsq MODE INPUT OUTPUT SHADER BLOCKS STEPS SAMPLES BITS SEED TARGET REPEATS");
    const bool rco_full=std::wcscmp(argv[1],L"rco_full")==0;
    const bool rco=rco_full || std::wcscmp(argv[1],L"rco")==0;
    const bool loss=std::wcscmp(argv[1],L"loss")==0;
    const bool actcal=std::wcscmp(argv[1],L"actcal")==0;
    const bool actcal_k=std::wcscmp(argv[1],L"actcal-k")==0;
    const bool flip=std::wcscmp(argv[1],L"macaq-flip")==0;
    const bool gsq_shared=std::wcscmp(argv[1],L"gsq_shared")==0;
    if (!rco && !loss && !actcal && !actcal_k && !flip && !gsq_shared && std::wcscmp(argv[1],L"gsq")!=0) throw std::runtime_error("unknown mode");
    wchar_t * target_end=nullptr;
    const float target=std::wcstof(argv[10],&target_end);
    if (target_end==argv[10] || *target_end || !std::isfinite(target)) throw std::runtime_error("invalid target");
    const unsigned block_limit=gsq_shared?16384:(rco_full?512:(actcal||flip?262144:(actcal_k?32768:(rco?8:4096))));
    const Parameters pc={integer(argv[5],block_limit),integer(argv[6],actcal||flip?8:128),integer(argv[7],actcal||flip?2048:(actcal_k?256:32)),
                         integer(argv[8],8),integer(argv[9],UINT32_MAX),target};
    const unsigned repeats=integer(argv[11],16);
    if (!pc.blocks || !pc.samples || (rco && ((pc.bits!=3 && pc.bits!=4) || pc.steps<2)) ||
        (rco_full && (pc.samples!=1 || pc.bits!=3 || pc.steps>32)) ||
        (gsq_shared && (target<1 || target>4096 || target!=std::floor(target) || pc.blocks%uint32_t(target)!=0)) ||
        (actcal && (pc.bits!=4 && pc.bits!=8)) ||
        (actcal_k && (pc.bits<1 || pc.bits>2 || pc.blocks%pc.samples!=0)) ||
        (flip && ((pc.bits!=4 && pc.bits!=5 && pc.bits!=6 && pc.bits!=8) || pc.blocks%pc.samples!=0)) ||
        (!rco && !loss && !actcal && !actcal_k && !flip && pc.bits!=2 && pc.bits!=3 && pc.bits!=4 && pc.bits!=6 && pc.bits!=8) ||
        repeats==0) throw std::runtime_error("parameters outside the supported range");
    const size_t input_floats=gsq_shared ? (size_t(pc.blocks)+size_t(target)*pc.samples)*256 :
        flip ? size_t(pc.blocks)*96+pc.samples*1024 : actcal_k ? (size_t(pc.blocks)*(pc.bits+1)+pc.samples)*256 :
        (actcal ? size_t(pc.blocks)*33+pc.samples*32 : (loss ? size_t(pc.blocks)*512 : size_t(pc.blocks)*(rco?pc.bits:256)*(1+pc.samples)));
    const size_t output_floats=flip ? size_t(pc.blocks)*8 : actcal_k ? size_t(pc.blocks)*pc.bits :
        (actcal ? size_t(pc.blocks)*(pc.bits==4?5:9) : (loss ? pc.blocks : (rco ? pc.blocks+4+pc.blocks*pc.bits : size_t(pc.blocks)*266)));
    const vk::DeviceSize input_size=input_floats*sizeof(float),output_size=output_floats*sizeof(float);
    std::ifstream input_file(std::filesystem::path(argv[2]),std::ios::binary|std::ios::ate);
    if (!input_file || input_file.tellg()!=std::streamoff(input_size)) throw std::runtime_error("input size mismatch");
    input_file.seekg(0);
    std::vector<float> input(input_floats);
    if (!input_file.read(reinterpret_cast<char *>(input.data()),std::streamsize(input_size)) ||
        !std::all_of(input.begin(),input.end(),[](float x){ return std::isfinite(x); })) throw std::runtime_error("invalid input floats");
    if (rco) {
        float min_cost=0, max_cost=0;
        for (uint32_t i=0;i<pc.blocks;++i) {
            const auto begin=input.begin()+i*pc.bits;
            min_cost+=*std::min_element(begin,begin+pc.bits);
            max_cost+=*std::max_element(begin,begin+pc.bits);
        }
        if (target<=min_cost || target>=max_cost) throw std::runtime_error("budget must lie strictly within the feasible continuous range");
    }

    const auto instance=vk::createInstanceUnique({{},&app});
    vk::PhysicalDevice physical;
    for (const auto & candidate:instance->enumeratePhysicalDevices()) {
        const auto p=candidate.getProperties();
        if (is_polaris(p)) {
            physical=candidate;
            break;
        }
    }
    if (!physical) throw std::runtime_error("No supported AMD Polaris Vulkan device found");
    const auto props=physical.getProperties();
    const auto families=physical.getQueueFamilyProperties();
    uint32_t family=0;
    while(family<families.size() && (!families[family].timestampValidBits ||
          !(families[family].queueFlags & vk::QueueFlagBits::eCompute))) ++family;
    if(family==families.size()) throw std::runtime_error("compute timestamps unavailable");
    constexpr float priority=1;
    const vk::DeviceQueueCreateInfo queue_info({},family,1,&priority);
    const auto device=physical.createDeviceUnique({{},1,&queue_info});
    const auto dev=*device;
    const auto memory=physical.getMemoryProperties();
    const auto queue=dev.getQueue(family,0);
    const auto pool=dev.createCommandPoolUnique({vk::CommandPoolCreateFlagBits::eResetCommandBuffer,family});
    const auto commands=dev.allocateCommandBuffersUnique({*pool,vk::CommandBufferLevel::ePrimary,1});
    const auto cmd=*commands[0];
    const auto queries=dev.createQueryPoolUnique({{},vk::QueryType::eTimestamp,2});
    constexpr auto transfer=vk::PipelineStageFlagBits::eTransfer;
    constexpr auto compute=vk::PipelineStageFlagBits::eComputeShader;
    constexpr auto local=vk::MemoryPropertyFlagBits::eDeviceLocal;
    constexpr auto io=vk::BufferUsageFlagBits::eTransferSrc|vk::BufferUsageFlagBits::eTransferDst;
    constexpr auto visible=vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent;
    const Buffer a=make_buffer(dev,memory,input_size,io|vk::BufferUsageFlagBits::eStorageBuffer,local);
    const Buffer b=make_buffer(dev,memory,output_size,io|vk::BufferUsageFlagBits::eStorageBuffer,local);
    const Buffer staged=make_buffer(dev,memory,std::max(input_size,output_size),io,visible);
    const auto map=dev.mapMemory(*staged.memory,0,std::max(input_size,output_size));
    std::memcpy(map,input.data(),input_size);
    const std::array<vk::DescriptorSetLayoutBinding,2> bindings={
        vk::DescriptorSetLayoutBinding(0,vk::DescriptorType::eStorageBuffer,1,vk::ShaderStageFlagBits::eCompute),
        vk::DescriptorSetLayoutBinding(1,vk::DescriptorType::eStorageBuffer,1,vk::ShaderStageFlagBits::eCompute),
    };
    const auto layout=dev.createDescriptorSetLayoutUnique({{},bindings});
    const vk::PushConstantRange range(vk::ShaderStageFlagBits::eCompute,0,sizeof(pc));
    const auto pl=dev.createPipelineLayoutUnique({{},*layout,range});
    const vk::DescriptorPoolSize storage(vk::DescriptorType::eStorageBuffer,2);
    const auto dpool=dev.createDescriptorPoolUnique({{},1,storage});
    const auto set=dev.allocateDescriptorSets({*dpool,*layout})[0];
    const std::array<vk::DescriptorBufferInfo,2> infos={
        vk::DescriptorBufferInfo(*a.buffer,0,input_size),
        vk::DescriptorBufferInfo(*b.buffer,0,output_size),
    };
    const vk::ArrayProxyNoTemporaries<const vk::DescriptorImageInfo> no_images;
    const std::array<vk::WriteDescriptorSet,2> writes={
        vk::WriteDescriptorSet(set,0,0,vk::DescriptorType::eStorageBuffer,no_images,infos[0]),
        vk::WriteDescriptorSet(set,1,0,vk::DescriptorType::eStorageBuffer,no_images,infos[1]),
    };
    dev.updateDescriptorSets(writes,{});
    const auto spv=shader(dev,argv[4]);
    const auto pipeline=dev.createComputePipelineUnique({},
        {{},vk::PipelineShaderStageCreateInfo({},vk::ShaderStageFlagBits::eCompute,*spv,"main"),*pl}).value;

    auto barrier=[&](vk::PipelineStageFlags before,vk::PipelineStageFlags after,
                     vk::AccessFlags writes,vk::AccessFlags reads) {
        cmd.pipelineBarrier(before,after,{},vk::MemoryBarrier(writes,reads),{},{});
    };
    auto submit=[&](){
        cmd.end();
        queue.submit(vk::SubmitInfo({}, {}, cmd, {}));
        queue.waitIdle();
    };
    cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    cmd.copyBuffer(*staged.buffer,*a.buffer,vk::BufferCopy(0,0,input_size));
    barrier(transfer,compute,vk::AccessFlagBits::eTransferWrite,vk::AccessFlagBits::eShaderRead);
    submit();
    std::cout << props.deviceName.data() << '\n';
    const unsigned warmups=repeats>1 ? 2 : 0;
    for (unsigned run=0;run<warmups+repeats;++run) {
        cmd.reset();
        cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        cmd.resetQueryPool(*queries,0,2);
        cmd.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe,*queries,0);
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute,*pipeline);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute,*pl,0,set,{});
        cmd.pushConstants(*pl,vk::ShaderStageFlagBits::eCompute,0,sizeof(pc),&pc);
        cmd.dispatch((actcal || actcal_k || flip)?(pc.blocks+63)/64:(rco?1:pc.blocks),1,1);
        cmd.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe,*queries,1);
        barrier(compute,transfer,vk::AccessFlagBits::eShaderWrite,vk::AccessFlagBits::eTransferRead);
        cmd.copyBuffer(*b.buffer,*staged.buffer,vk::BufferCopy(0,0,output_size));
        submit();
        std::array<uint64_t,2> values{};
        const auto status=dev.getQueryPoolResults(*queries,0,2,sizeof(values),values.data(),8,
            vk::QueryResultFlagBits::e64|vk::QueryResultFlagBits::eWait);
        if (status!=vk::Result::eSuccess) throw std::runtime_error("timestamp queries failed");
        const uint64_t mask= families[family].timestampValidBits==64 ?
            std::numeric_limits<uint64_t>::max() : ((uint64_t(1)<<families[family].timestampValidBits)-1);
        if (run>=warmups) std::cout << ((values[1]-values[0])&mask)*props.limits.timestampPeriod/1000.0 << '\n';
    }
    std::ofstream output(std::filesystem::path(argv[3]),std::ios::binary);
    if (!output.write(static_cast<const char *>(map),std::streamsize(output_size))) throw std::runtime_error("could not write the result");
    dev.unmapMemory(*staged.memory);
    return 0;
} catch(const std::exception & error) {
    std::cerr << error.what() << '\n';
    return 1;
}
