using System.Collections.Generic;
using System.Linq;
using SabberStoneCore.Model;

namespace SabberStoneEnv;

/// <summary>
/// Stable card-ID vocabulary: Card -> dense index, for the policy's card embedding.
///
/// Every env process must agree on these indices, otherwise tokens from different envs mean
/// different things and the embedding learns noise. `Cards.All` is a Dictionary's Values, whose
/// enumeration order carries no ordering guarantee, so the vocabulary is built by sorting on the
/// string card Id ("CS2_029") with an ordinal comparison — stable across processes, runs, and
/// framework versions.
///
/// Index 0 is reserved for "no card / unknown", so a missing lookup degrades to a learned
/// unknown-token embedding rather than colliding with a real card.
/// </summary>
public static class CardVocab
{
    private static readonly Dictionary<string, int> Index;

    /// <summary>Number of embedding rows required (including the reserved 0 slot).</summary>
    public static int Count { get; }

    static CardVocab()
    {
        string[] ids = Cards.All
            .Select(c => c.Id)
            .Where(id => !string.IsNullOrEmpty(id))
            .Distinct()
            .OrderBy(id => id, System.StringComparer.Ordinal)
            .ToArray();

        Index = new Dictionary<string, int>(ids.Length);
        for (int i = 0; i < ids.Length; i++)
            Index[ids[i]] = i + 1;   // 0 reserved

        Count = ids.Length + 1;
    }

    /// <summary>Dense index for a card; 0 when null or unknown.</summary>
    public static int IndexOf(Card? card) =>
        card != null && card.Id != null && Index.TryGetValue(card.Id, out int i) ? i : 0;
}
