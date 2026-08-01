using System;
using System.Collections.Generic;
using System.Linq;
using SabberStoneCore.Enums;
using SabberStoneCore.Model;

namespace SabberStoneEnv;

/// <summary>
/// Random-but-legal 30-card decks, for the varied-deck training/eval modes.
///
/// Why this exists rather than just turning `FillDecks` loose: SabberStone's fill gives each
/// player an independently random pile, so one side routinely draws a materially stronger deck
/// than the other. Win rate then measures *deck luck* on top of policy strength, which is the
/// last thing you want when the number you are chasing is a few points wide. <see cref="Mirror"/>
/// hands both players the identical list, so the matchup stays fair while the card pool still
/// varies game to game — that is the setting where card semantics can actually pay off, since a
/// fixed mirror has no unseen cards to generalise to.
/// </summary>
public static class DeckBuilder
{
    private const int DeckSize = 30;

    /// <summary>
    /// Legal deck pool for a class: collectible Standard cards SabberStone actually implements.
    ///
    /// The `Implemented` filter is not optional. `Cards.Standard[cls]` includes cards whose
    /// effects are unwritten; drawing one mid-game throws, which would surface as sporadic env
    /// crashes on a small fraction of games rather than an obvious failure at startup.
    /// </summary>
    private static readonly Dictionary<CardClass, Card[]> Pools = new();

    private static Card[] PoolFor(CardClass cls)
    {
        lock (Pools)
        {
            if (Pools.TryGetValue(cls, out Card[]? cached)) return cached;

            Card[] pool = Cards.Standard[cls]
                .Where(c => c.Implemented && c.Type != CardType.HERO)
                // Ordinal sort for the same reason CardVocab sorts: `Standard` is built from
                // `Cards.All` enumeration order, so an unsorted pool would make identical seeds
                // produce different decks across processes.
                .OrderBy(c => c.Id, StringComparer.Ordinal)
                .ToArray();

            Pools[cls] = pool;
            return pool;
        }
    }

    /// <summary>
    /// A 30-card deck for <paramref name="cls"/>, drawn with the standard copy limits
    /// (1 legendary / 2 otherwise). Deterministic given <paramref name="rnd"/>.
    /// </summary>
    public static List<Card> Build(CardClass cls, Random rnd)
    {
        Card[] pool = PoolFor(cls);
        var counts = new Dictionary<string, int>();
        var deck = new List<Card>(DeckSize);

        // Rejection sampling rather than a shuffle: the copy limit means a plain shuffle would
        // need reshuffling anyway, and the pool is ~2 orders of magnitude larger than the deck so
        // rejections are rare. The attempt cap stops a pathologically small pool from hanging.
        int attempts = 0;
        while (deck.Count < DeckSize && attempts++ < DeckSize * 100)
        {
            Card c = pool[rnd.Next(pool.Length)];
            int limit = c.Rarity == Rarity.LEGENDARY ? 1 : 2;
            counts.TryGetValue(c.Id, out int have);
            if (have >= limit) continue;
            counts[c.Id] = have + 1;
            deck.Add(c);
        }

        return deck;
    }

    /// <summary>
    /// One deck, shared by both players. Returns the class alongside it because both seats must
    /// also be that class for the deck to be legal for them.
    /// </summary>
    public static (CardClass Class, List<Card> Deck) Mirror(CardClass[] classes, Random rnd)
    {
        CardClass cls = classes[rnd.Next(classes.Length)];
        return (cls, Build(cls, rnd));
    }
}
