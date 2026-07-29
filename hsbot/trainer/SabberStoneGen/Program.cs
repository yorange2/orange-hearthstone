using SabberStoneGen;

// Usage: dotnet run -c Release -- [numGames=1000] [outPath=data/selfplay.jsonl] [seed=1]
int numGames = args.Length > 0 ? int.Parse(args[0]) : 1000;
string outPath = args.Length > 1 ? args[1] : Path.Combine("data", "selfplay.jsonl");
int seed = args.Length > 2 ? int.Parse(args[2]) : 1;

Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outPath))!);

Console.Error.WriteLine($"self-play: {numGames} games, seed={seed} -> {outPath}");
var sw = System.Diagnostics.Stopwatch.StartNew();

using var writer = new StreamWriter(outPath);
long rows = new SelfPlayGenerator().Generate(numGames, seed, writer);

Console.Error.WriteLine($"done: {rows} rows in {sw.Elapsed.TotalSeconds:F1}s " +
                        $"({numGames / Math.Max(sw.Elapsed.TotalSeconds, 1e-9):F0} games/s)");
