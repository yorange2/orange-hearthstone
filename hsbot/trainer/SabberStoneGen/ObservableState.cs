namespace SabberStoneGen;

/// <summary>
/// Engine-neutral observable board state — the data structure the feature contract is
/// defined over (see hsbot/docs/FEATURES.md, hsbot/fixtures/README.md). Deserializes
/// directly from a fixture's `state` object, and is produced from a live game by
/// <see cref="SabberStoneObserver"/>. Observable-only: opponent hand/deck are counts.
/// </summary>
public sealed class ObservableState
{
    public int Turn { get; set; }
    public bool MeIsFirst { get; set; }
    public Side Me { get; set; } = new();
    public Side Opp { get; set; } = new();
}

public sealed class Side
{
    public HeroInfo Hero { get; set; } = new();
    public ManaInfo Mana { get; set; } = new();
    public int Hand { get; set; }
    public int Deck { get; set; }
    public List<MinionInfo> Board { get; set; } = new();
    public int Secrets { get; set; }
    public WeaponInfo? Weapon { get; set; }
    public int Fatigue { get; set; }
}

public sealed class HeroInfo
{
    /// <summary>Remaining HP including armor (health + armor).</summary>
    public int EffectiveHp { get; set; }
    /// <summary>Remaining armor.</summary>
    public int Armor { get; set; }
}

public sealed class ManaInfo
{
    /// <summary>Max crystals this turn.</summary>
    public int Max { get; set; }
    public int Overload { get; set; }
}

public sealed class WeaponInfo
{
    public int Attack { get; set; }
    /// <summary>Remaining durability.</summary>
    public int Durability { get; set; }
}

public sealed class MinionInfo
{
    public int Attack { get; set; }
    public int Health { get; set; }
    public bool CanAttack { get; set; }
    public bool Taunt { get; set; }
    public bool DivineShield { get; set; }
    public bool Windfury { get; set; }
    public bool Lifesteal { get; set; }
    public bool Poisonous { get; set; }
}
