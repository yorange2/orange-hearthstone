using System.Text.Json;
using SabberStoneEnv;

// Line-based stdio env server (one env per process). The Python PPO learner drives it:
//   stdin:  "reset"          stdout: {"tokens":[[18]...],"priv":[8],"actions":[[20]...],"reward":0,"done":false}
//           "step <index>"   stdout: {"tokens":[...],"priv":[8],"actions":[...],"reward":r,"done":bool}
// On done, tokens/actions are empty; the driver then sends "reset". `priv` is opponent-hidden
// info for the critic only. A simple bridge; swap for gRPC when scaling to many parallel actors.

int seed = args.Length > 0 ? int.Parse(args[0]) : 1;
var env = new HearthstoneEnv(seed);
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
        Write(env.Tokens, env.Privileged, env.LegalActionFeatures, 0f, false);
    }
    else if (line.StartsWith("step"))
    {
        int idx = int.Parse(line.AsSpan(4).Trim());
        var (reward, done) = env.Step(idx);
        Write(env.Tokens, env.Privileged, env.LegalActionFeatures, reward, done);
    }
    else if (line == "close")
    {
        break;
    }
}

void Write(float[][] tokens, float[] priv, float[][] actions, float reward, bool done)
{
    var resp = new Response { Tokens = tokens, Priv = priv, Actions = actions, Reward = reward, Done = done };
    stdout.WriteLine(JsonSerializer.Serialize(resp, jsonOpts));
    stdout.Flush();
}

sealed class Response
{
    public float[][] Tokens { get; set; } = System.Array.Empty<float[]>();
    public float[] Priv { get; set; } = System.Array.Empty<float>();
    public float[][] Actions { get; set; } = System.Array.Empty<float[]>();
    public float Reward { get; set; }
    public bool Done { get; set; }
}
