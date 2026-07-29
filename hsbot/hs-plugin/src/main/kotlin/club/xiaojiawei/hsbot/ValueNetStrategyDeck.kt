package club.xiaojiawei.hsbot

import club.xiaojiawei.hsscriptbase.enums.RunModeEnum
import club.xiaojiawei.hsscriptcardsdk.bean.Card
import club.xiaojiawei.hsscriptcardsdk.bean.MCTSArg
import club.xiaojiawei.hsscriptcardsdk.bean.ScoreCalculator
import club.xiaojiawei.hsscriptcardsdk.bean.War
import club.xiaojiawei.hsscriptstrategysdk.deck.MCTSDeckStrategy
import java.nio.file.Path

/**
 * A deck strategy that drives HS-Script's built-in MCTS with a learned value net instead
 * of the hand-tuned heuristic. We only supply the [ScoreCalculator]; the framework does
 * action enumeration, simulation, search, and execution on the live client.
 *
 * See hsbot/docs/FEATURES.md for the state contract the net was trained against.
 */
class ValueNetStrategyDeck : MCTSDeckStrategy() {

    // Lazily resolved so a missing/broken model degrades to the heuristic (see OnnxValueNet).
    private val scoreCalculator: ScoreCalculator by lazy {
        OnnxValueNet.loadOrHeuristic(MODEL_PATH)
    }

    override fun name(): String = "ValueNet (v1)"

    override fun description(): String = "MCTS guided by an ONNX value net trained on SabberStone self-play."

    override fun getRunMode(): Array<RunModeEnum> = RUN_MODES

    // TODO: paste the real deck code for the deck this strategy plays. Must match the deck
    //  you generate self-play data for, or the value net sees an off-distribution game.
    override fun deckCode(): String = "PASTE_DECK_CODE_HERE"

    override fun id(): String = "orange-hearthstone.valuenet.v1"

    override fun executeMCTSOutCard(war: War): List<MCTSArg> = listOf(
        MCTSArg(
            /* endMillisTime   */ System.currentTimeMillis() + THINK_MILLIS,
            /* turnCount       */ 2,      // this turn + 1 opponent look-ahead (framework's "inverse")
            /* turnFactor      */ 0.8,    // discount the opponent's best response
            /* countPerTurn    */ 3000,   // simulations budget; tune against THINK_MILLIS
            /* scoreCalculator */ scoreCalculator,
            /* enableMultiThread*/ true,
        ),
    )

    /**
     * Mulligan. Placeholder heuristic: toss anything over 3 mana. Replace with a learned
     * mulligan policy or per-deck keep rules (a separate small model, per FEATURES.md §5).
     */
    override fun executeChangeCard(cards: HashSet<Card>) {
        cards.toList().forEach { if (it.cost <= 3) cards.remove(it) }
    }

    /**
     * Discover. Placeholder: pick the first option. Could be scored with the value net by
     * simulating each choice, but keep simple until the main policy is validated.
     */
    override fun executeDiscoverChooseCard(vararg cards: Card): Int = 0

    companion object {
        private val RUN_MODES = arrayOf(RunModeEnum.STANDARD)
        private const val THINK_MILLIS = 4000L

        // Ship the trained model next to the plugin; make this configurable later.
        private val MODEL_PATH: Path =
            Path.of(System.getProperty("user.home"), ".hs-script", "models", "value_net.v1.onnx")
    }
}
