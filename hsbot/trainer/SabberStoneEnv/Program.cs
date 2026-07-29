using System.Text.Json;
using SabberStoneEnv;

// Line-based stdio env server (one env per process), seat-agnostic for self-play. The Python
// driver routes each decision to the main policy or a league opponent by `player`:
//   stdin:  "reset"          stdout: {"tokens":[[18]...],"priv":[8],"actions":[[20]...],"player":p,"done":false,"winner":0}
//           "step <index>"   stdout: {"tokens":[...],"priv":[8],"actions":[...],"player":p,"done":bool,"winner":w}
// `player` (1/2) is the current mover; on done tokens/actions are empty and `winner` is 1/2/0.
// `priv` is the current mover's opponent-hidden info (critic-only). Swap stdio for gRPC to scale.

int seed = args.Length > 0 ? int.Parse(args[0]) : 1;
bool fixedDeck = args.Length > 1 && args[1] == "1";
var env = new HearthstoneEnv(seed, fixedDeck);
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
    else if (line == "step_greedy")
    {
        bool done = env.StepGreedy();  // env plays the current mover's greedy heuristic action
        Write(env, done);
    }
    else if (line.StartsWith("step"))
    {
        int idx = int.Parse(line.AsSpan(4).Trim());
        bool done = env.Step(idx);
        Write(env, done);
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
        Tokens = e.Tokens, Priv = e.Privileged, Actions = e.LegalActionFeatures,
        Player = e.CurrentPlayerId, Potential = e.Potential, Done = done, Winner = e.Winner,
    };
    stdout.WriteLine(JsonSerializer.Serialize(resp, jsonOpts));
    stdout.Flush();
}

sealed class Response
{
    public float[][] Tokens { get; set; } = System.Array.Empty<float[]>();
    public float[] Priv { get; set; } = System.Array.Empty<float>();
    public float[][] Actions { get; set; } = System.Array.Empty<float[]>();
    public int Player { get; set; }
    public float Potential { get; set; }
    public bool Done { get; set; }
    public int Winner { get; set; }
}
