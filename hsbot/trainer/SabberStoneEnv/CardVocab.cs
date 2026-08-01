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
    private static readonly Card?[] ByIndex;

    /// <summary>Number of embedding rows required (including the reserved 0 slot).</summary>
    public static int Count { get; }

    static CardVocab()
    {
        // One Card per Id, chosen deterministically: several entries can share an Id, and taking
        // "whichever the dictionary yielded first" would make Name/Text differ across processes
        // even though the indices agree — a silent way for the text-embedding matrix built by one
        // process to mismatch the tokens emitted by another.
        Dictionary<string, Card> byId = new();
        foreach (Card c in Cards.All)
        {
            if (string.IsNullOrEmpty(c.Id)) continue;
            if (!byId.ContainsKey(c.Id)) byId[c.Id] = c;
        }

        string[] ids = byId.Keys.OrderBy(id => id, System.StringComparer.Ordinal).ToArray();

        Index = new Dictionary<string, int>(ids.Length);
        ByIndex = new Card?[ids.Length + 1];
        for (int i = 0; i < ids.Length; i++)
        {
            Index[ids[i]] = i + 1;   // 0 reserved
            ByIndex[i + 1] = byId[ids[i]];
        }

        Count = ids.Length + 1;
    }

    /// <summary>
    /// Fingerprint of the id list, in index order. A card-text matrix records the same value at
    /// build time, so pairing it with an env whose pool has since changed is rejected rather than
    /// silently reading shifted rows. Deliberately matches `card_text.ids_hash` on the Python
    /// side: sha256 over the ids joined by "\n" (including the empty row 0), first 16 hex chars.
    /// </summary>
    public static string IdsHash()
    {
        var sb = new System.Text.StringBuilder();
        for (int i = 0; i < Count; i++)
        {
            if (i > 0) sb.Append('\n');
            sb.Append(i == 0 ? "" : ByIndex[i]?.Id ?? "");
        }
        byte[] hash = System.Security.Cryptography.SHA256.HashData(
            System.Text.Encoding.UTF8.GetBytes(sb.ToString()));
        return System.Convert.ToHexString(hash).ToLowerInvariant()[..16];
    }

    /// <summary>Dense index for a card; 0 when null or unknown.</summary>
    public static int IndexOf(Card? card) =>
        card != null && card.Id != null && Index.TryGetValue(card.Id, out int i) ? i : 0;

    /// <summary>
    /// Id / name / rules text for every row, in index order (row 0 is the reserved blank).
    ///
    /// This exists so the text-embedding matrix is built from the *same* process that assigns the
    /// indices. Re-deriving the ordering on the Python side from CardDefs.xml would work right up
    /// until the two card pools diverge by one entry, after which every row past that point is
    /// shifted and the model silently reads the wrong card's meaning.
    /// </summary>
    public static (string Id, string Name, string Text)[] Rows()
    {
        var rows = new (string, string, string)[Count];
        rows[0] = ("", "", "");
        for (int i = 1; i < Count; i++)
        {
            Card? c = ByIndex[i];
            rows[i] = (c?.Id ?? "", c?.Name ?? "", c?.Text ?? "");
        }
        return rows;
    }
}
