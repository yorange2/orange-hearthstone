using SabberStoneCore.Model;
using SabberStoneCore.Model.Entities;

namespace SabberStoneGen;

/// <summary>
/// Produces the engine-neutral <see cref="ObservableState"/> from a live SabberStone game,
/// from the current player's perspective (the player whose turn is being optimized).
///
/// Observable-only per contract: opponent hand/deck are counts (we never read their card
/// identities); the opponent board and secrets ARE observable and included. This mirrors
/// what HS-Script's <c>War</c> knows at runtime.
///
/// Accessor mapping is documented field-by-field in hsbot/docs/FEATURES.md §4.
/// </summary>
public sealed class SabberStoneObserver
{
    public ObservableState Observe(Game game)
    {
        Controller me = game.CurrentPlayer;
        Controller opp = game.CurrentOpponent;
        return new ObservableState
        {
            Turn = game.Turn, // NOTE: SabberStone counts per player-turn; see FEATURES.md §4A
            MeIsFirst = me.PlayerId == game.FirstPlayer.PlayerId,
            Me = ToSide(me),
            Opp = ToSide(opp),
        };
    }

    private static Side ToSide(Controller c)
    {
        Hero hero = c.Hero;
        var board = new List<MinionInfo>(c.BoardZone.Count);
        foreach (Minion m in c.BoardZone)
        {
            board.Add(new MinionInfo
            {
                Attack = m.AttackDamage,
                Health = m.Health,
                CanAttack = m.CanAttack,
                Taunt = m.HasTaunt,
                DivineShield = m.HasDivineShield,
                Windfury = m.HasWindfury,
                Lifesteal = m.HasLifeSteal,
                Poisonous = m.Poisonous,
            });
        }

        return new Side
        {
            Hero = new HeroInfo { EffectiveHp = hero.Health + hero.Armor, Armor = hero.Armor },
            Mana = new ManaInfo { Max = c.BaseMana, Overload = c.OverloadLocked },
            Hand = c.HandZone.Count,
            Deck = c.DeckZone.Count,
            Board = board,
            Secrets = c.SecretZone.Count,
            Weapon = hero.Weapon == null
                ? null
                : new WeaponInfo { Attack = hero.Weapon.AttackDamage, Durability = hero.Weapon.Durability },
            Fatigue = hero.Fatigue,
        };
    }
}
