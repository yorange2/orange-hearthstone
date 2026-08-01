using System.Text.Json;
using SabberStoneEnv;

// Line-based stdio env server (one env per process), seat-agnostic for self-play. The Python
// driver routes each decision to the main policy or a league opponent by `player`:
//   stdin:  "reset"          stdout: {"tokens":[[18]...],"priv":[8],"actions":[[20]...],"player":p,"done":false,"winner":0}
//           "step <index>"   stdout: {"tokens":[...],"priv":[8],"actions":[...],"player":p,"done":bool,"winner":w}
// `player` (1/2) is the current mover; on done tokens/actions are empty and `winner` is 1/2/0.
// `priv` is the current mover's opponent-hidden info (critic-only). Swap stdio for gRPC to scale.

int seed = args.Length > 0 ? int.Parse(args[0]) : 1;
// args[1]: deck mode. Accepts the historical "1"/"0" (fixed / random) so existing callers keep
// working, plus the named modes ("fixed" | "variedmirror" | "random") — see DeckMode.
var deck = args.Length > 1
    ? args[1] switch
    {
        "1" => HearthstoneEnv.DeckMode.Fixed,
        "0" => HearthstoneEnv.DeckMode.Random,
        var s => System.Enum.Parse<HearthstoneEnv.DeckMode>(s, ignoreCase: true),
    }
    : HearthstoneEnv.DeckMode.Random;
// args[2]: shaping-potential mode ("midrange" default | "board" | "none") — see PotentialMode.
var potential = args.Length > 2
    ? System.Enum.Parse<HearthstoneEnv.PotentialMode>(args[2], ignoreCase: true)
    : HearthstoneEnv.PotentialMode.MidRange;
var env = new HearthstoneEnv(seed, deck, potential);
var jsonOpts = new JsonSerializerOptions
{
    PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
    NumberHandling = System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals,
};

var stdout = Console.Out;
string? line;
while ((line = Console.In.ReadLine()) != null)
{
    line = line.Trim();
    if (line.Length == 0) continue;

    if (line == "reset")
    {
        env.Reset();
        Write(env, false);
    }
    else if (line.StartsWith("step_greedy"))
    {
        // "step_greedy"         → midrange (default)
        // "step_greedy aggro"   → AggroScore, etc.
        string strategy = line.Length > 12 ? line[12..].Trim() : "midrange";
        bool done = env.StepGreedy(strategy);
        Write(env, done);
    }
    else if (line.StartsWith("step "))
    {
        int idx = int.Parse(line.AsSpan(4).Trim());
        bool done = env.Step(idx);
        Write(env, done);
    }
    else if (line == "simulate")
    {
        // Phase 3: resulting state per legal action, for critic-scored one-ply lookahead.
        // Read-only w.r.t. the live game — it clones.
        stdout.WriteLine(JsonSerializer.Serialize(new SimResponse { States = env.SimulateAll() }, jsonOpts));
        stdout.Flush();
    }
    else if (line == "meta")
    {
        // Static env facts the Python side needs before building the model — currently the
        // card-embedding vocabulary size. Queried once at startup, so it must not touch game state.
        stdout.WriteLine(JsonSerializer.Serialize(new Meta { CardVocab = CardVocab.Count, CardIdsHash = CardVocab.IdsHash() }, jsonOpts));
        stdout.Flush();
    }
    else if (line == "cards")
    {
        // Id / name / rules text for every vocabulary row, in index order, so the Python side can
        // build a card-text embedding matrix that is aligned with these indices by construction
        // rather than by a re-derivation that can drift. Static; does not touch game state.
        var rows = CardVocab.Rows();
        var cards = new CardRow[rows.Length];
        for (int i = 0; i < rows.Length; i++)
            cards[i] = new CardRow { Id = rows[i].Id, Name = rows[i].Name, Text = rows[i].Text };
        stdout.WriteLine(JsonSerializer.Serialize(new CardsResponse { Cards = cards }, jsonOpts));
        stdout.Flush();
    }
    else if (line == "close")
    {
        break;
    }
}

void Write(HearthstoneEnv e, bool done)
{
    var resp = new Response
    {
        Tokens = e.Tokens, CardIds = e.CardIds, Flat = e.Flat, Priv = e.Privileged,
        Actions = e.LegalActionFeatures,
        Player = e.CurrentPlayerId, Potential = e.Potential, Done = done, Winner = e.Winner,
        GreedyAction = e.GreedyActionIndex,
    };
    stdout.WriteLine(JsonSerializer.Serialize(resp, jsonOpts));
    stdout.Flush();
}

sealed class Response
{
    public float[][] Tokens { get; set; } = System.Array.Empty<float[]>();
    public int[] CardIds { get; set; } = System.Array.Empty<int>();
    public float[] Flat { get; set; } = System.Array.Empty<float>();
    public float[] Priv { get; set; } = System.Array.Empty<float>();
    public float[][] Actions { get; set; } = System.Array.Empty<float[]>();
    public int Player { get; set; }
    public float Potential { get; set; }
    public bool Done { get; set; }
    public int Winner { get; set; }
    public int GreedyAction { get; set; }
}

sealed class Meta
{
    public int CardVocab { get; set; }
    public string CardIdsHash { get; set; } = "";
}

sealed class CardRow
{
    public string Id { get; set; } = "";
    public string Name { get; set; } = "";
    public string Text { get; set; } = "";
}

sealed class CardsResponse
{
    public CardRow[] Cards { get; set; } = System.Array.Empty<CardRow>();
}

sealed class SimResponse
{
    public HearthstoneEnv.SimState[] States { get; set; } = System.Array.Empty<HearthstoneEnv.SimState>();
}
