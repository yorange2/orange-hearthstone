using System.Text.Json;
using SabberStoneGen;
using Xunit;

namespace SabberStoneGen.Tests;

/// <summary>
/// Parity guard for the C# extractor: it must reproduce every fixture's `expected` vector
/// within 1e-6 — the same fixtures the Kotlin and Python implementations are checked against,
/// keeping all three in lockstep.
/// </summary>
public class FeatureParityTests
{
    private static readonly JsonSerializerOptions Opts = new() { PropertyNameCaseInsensitive = true };

    public static IEnumerable<object[]> Fixtures()
    {
        foreach (var file in Directory.GetFiles(FixturesDir(), "*.json").OrderBy(f => f))
            yield return new object[] { file };
    }

    [Theory]
    [MemberData(nameof(Fixtures))]
    public void Extractor_matches_fixture(string file)
    {
        var doc = JsonSerializer.Deserialize<FixtureDoc>(File.ReadAllText(file), Opts)!;
        Assert.Equal("v1", doc.Version);

        float[] actual = FeatureExtractor.Extract(doc.State);

        Assert.Equal(doc.Expected.Length, actual.Length);
        for (int i = 0; i < doc.Expected.Length; i++)
        {
            Assert.True(
                Math.Abs(doc.Expected[i] - actual[i]) <= 1e-6f,
                $"feature[{i}] mismatch in {Path.GetFileName(file)}: expected {doc.Expected[i]}, got {actual[i]}");
        }
    }

    private static string FixturesDir()
    {
        // Walk up from the test bin dir to find <repo>/hsbot/fixtures.
        var dir = AppContext.BaseDirectory;
        for (int i = 0; i < 12 && dir is not null; i++)
        {
            var candidate = Path.Combine(dir, "hsbot", "fixtures");
            if (Directory.Exists(candidate)) return candidate;
            dir = Directory.GetParent(dir)?.FullName;
        }
        throw new DirectoryNotFoundException("could not locate hsbot/fixtures from " + AppContext.BaseDirectory);
    }

    private sealed class FixtureDoc
    {
        public string Version { get; set; } = "";
        public ObservableState State { get; set; } = new();
        public float[] Expected { get; set; } = Array.Empty<float>();
    }
}
