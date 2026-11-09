using System;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Reflection;
using System.Text;
using System.Windows.Forms;

[assembly: AssemblyTitle("GGUF Converter")]
[assembly: AssemblyProduct("GGUF Converter")]
[assembly: AssemblyVersion("1.0.0.0")]

internal static class Launcher
{
    // Match the Windows argv rules for backslashes immediately before quotes.
    private static string Quote(string value)
    {
        var result = new StringBuilder("\"");
        int slashes = 0;
        foreach (char c in value)
        {
            if (c == '\\') { slashes++; continue; }
            result.Append('\\', c == '"' ? 2 * slashes + 1 : slashes);
            result.Append(c);
            slashes = 0;
        }
        return result.Append('\\', 2 * slashes).Append('"').ToString();
    }

    private static void Extract(string root)
    {
        string prefix = root + Path.DirectorySeparatorChar;
        using (Stream payload = Assembly.GetExecutingAssembly().GetManifestResourceStream("converter.zip"))
        using (var archive = new ZipArchive(payload, ZipArchiveMode.Read))
        {
            foreach (ZipArchiveEntry entry in archive.Entries)
            {
                if (Path.IsPathRooted(entry.FullName) || entry.FullName.Contains(":"))
                    throw new IOException("Invalid archive path.");
                string target = Path.GetFullPath(Path.Combine(root, entry.FullName));
                if (!target.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
                    throw new IOException("Archive path leaves the temporary folder.");
                Directory.CreateDirectory(Path.GetDirectoryName(target));
                using (Stream input = entry.Open())
                using (var output = new FileStream(target, FileMode.CreateNew, FileAccess.Write))
                {
                    input.CopyTo(output);
                    if (output.Length != entry.Length)
                        throw new IOException("Incomplete archive entry: " + entry.FullName);
                }
            }
        }
    }

    [STAThread]
    private static int Main(string[] args)
    {
        bool cli = args.Length > 0 && args[0] == "--cli";
        string scratch = Path.GetFullPath(Path.Combine(Path.GetTempPath(), "."));
        string root = Path.Combine(scratch, "gguf-" + Guid.NewGuid().ToString("N").Substring(0, 16));
        bool owned = false;
        try
        {
            if (Directory.Exists(root) || File.Exists(root))
                throw new IOException("The temporary folder already exists.");
            Directory.CreateDirectory(root);
            owned = true;
            Extract(root);

            string app = Path.Combine(root, "app");
            string python = Path.Combine(root, "python");
            var arguments = new StringBuilder("-I -B -X utf8 " + Quote(Path.Combine(app, "convert.py")));
            if (!cli) arguments.Append(" --gui");
            for (int i = cli ? 1 : 0; i < args.Length; i++)
                arguments.Append(' ').Append(Quote(args[i]));

            var start = new ProcessStartInfo(Path.Combine(python, "python.exe"), arguments.ToString());
            start.UseShellExecute = false;
            start.CreateNoWindow = true;
            start.WorkingDirectory = app;
            start.RedirectStandardError = true;
            start.EnvironmentVariables["GGUF_CONVERTER_CWD"] = Environment.CurrentDirectory;
            start.EnvironmentVariables["PATH"] = python + ";" + Environment.SystemDirectory;
            foreach (string name in new[] { "PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "VIRTUAL_ENV", "TCL_LIBRARY", "TK_LIBRARY" })
                start.EnvironmentVariables.Remove(name);
            using (Process child = Process.Start(start))
            {
                string error = child.StandardError.ReadToEnd();
                child.WaitForExit();
                if (child.ExitCode != 0)
                    throw new InvalidOperationException(error.Length > 0 ? error.Trim() : "Converter exited with code " + child.ExitCode);
            }
            return 0;
        }
        catch (Exception error)
        {
            if (cli) Console.Error.WriteLine(error.Message);
            else MessageBox.Show(error.Message, "GGUF Converter", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 1;
        }
        finally
        {
            // Only the freshly created, uniquely named extraction directory is ours.
            if (owned && Directory.Exists(root) &&
                String.Equals(Path.GetDirectoryName(Path.GetFullPath(root)), scratch, StringComparison.OrdinalIgnoreCase) &&
                (File.GetAttributes(root) & FileAttributes.ReparsePoint) == 0)
            {
                try { Directory.Delete(root, true); }
                catch (IOException) { }
                catch (UnauthorizedAccessException) { }
            }
        }
    }
}
