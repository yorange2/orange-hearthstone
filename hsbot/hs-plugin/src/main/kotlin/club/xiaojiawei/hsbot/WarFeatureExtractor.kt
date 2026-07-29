package club.xiaojiawei.hsbot

import club.xiaojiawei.hsscriptcardsdk.bean.Player
import club.xiaojiawei.hsscriptcardsdk.bean.War

/**
 * Runtime half of the shared feature contract (see hsbot/docs/FEATURES.md, v1).
 *
 * Produces a fixed-length 144 float vector from an HS-Script [War], from the perspective
 * of `war.me` (the player whose turn is being optimized). MUST stay byte-identical to the
 * C# `POGameFeatureExtractor`; both are checked against hsbot/fixtures/*.json.
 *
 * Any change here is a contract change: bump FEATURES.md version and regenerate fixtures.
 */
object WarFeatureExtractor {

    const val VERSION = "v1"
    const val LENGTH = 144

    // Section offsets
    private const val GLOBAL_PER_PLAYER = 15
    private const val TURN_BASE = 30          // features 30..31
    private const val MINION_SECTION_BASE = 32
    private const val MINION_SLOTS = 7
    private const val MINION_FEATS = 8
    private const val MINION_PER_PLAYER = MINION_SLOTS * MINION_FEATS // 56

    fun extract(war: War): FloatArray {
        val f = FloatArray(LENGTH)
        val me = war.me
        val rival = war.rival

        // 4A. global scalars — [me, opp]
        writeGlobals(f, 0 * GLOBAL_PER_PLAYER, me)
        writeGlobals(f, 1 * GLOBAL_PER_PLAYER, rival)

        // turn-level
        f[TURN_BASE + 0] = war.warTurn / 30f            // see FEATURES.md §4A turn-semantics note
        f[TURN_BASE + 1] = if (me.gameId == war.firstPlayerGameId) 1f else 0f

        // 4B. per-minion slots — [me, opp]
        writeMinions(f, MINION_SECTION_BASE + 0 * MINION_PER_PLAYER, me)
        writeMinions(f, MINION_SECTION_BASE + 1 * MINION_PER_PLAYER, rival)

        return f
    }

    private fun writeGlobals(f: FloatArray, base: Int, p: Player) {
        val play = p.playArea
        val hero = play.hero
        val weapon = play.weapon
        val board = play.cards

        val heroEffHp = hero?.blood() ?: 0
        val heroArmorRemaining = hero?.let { maxOf(it.armor - it.damage, 0) } ?: 0

        f[base + 0] = heroEffHp / 30f
        f[base + 1] = heroArmorRemaining / 30f
        f[base + 2] = p.handArea.cards.size / 10f
        f[base + 3] = p.deckArea.cardSize() / 30f
        f[base + 4] = board.size / 7f
        f[base + 5] = board.sumOf { it.atc } / 30f
        f[base + 6] = board.sumOf { it.blood() } / 40f
        f[base + 7] = board.count { it.isTaunt } / 7f
        f[base + 8] = board.count { it.isDivineShield } / 7f
        f[base + 9] = p.secretArea.cards.size / 5f
        f[base + 10] = (weapon?.atc ?: 0) / 10f
        f[base + 11] = (weapon?.blood() ?: 0) / 5f
        f[base + 12] = p.resources / 10f
        f[base + 13] = p.overloadLocked / 5f
        f[base + 14] = p.fatigue / 10f
    }

    private fun writeMinions(f: FloatArray, base: Int, p: Player) {
        val board = p.playArea.cards
        var i = 0
        while (i < MINION_SLOTS && i < board.size) {
            val c = board[i]
            val o = base + i * MINION_FEATS
            f[o + 0] = c.atc / 12f
            f[o + 1] = c.blood() / 12f
            f[o + 2] = b2f(c.canAttack())
            f[o + 3] = b2f(c.isTaunt)
            f[o + 4] = b2f(c.isDivineShield)
            f[o + 5] = b2f(c.isWindFury)
            f[o + 6] = b2f(c.isLifesteal)
            f[o + 7] = b2f(c.isPoisonous)
            i++
        }
        // remaining slots already zero-padded
    }

    private fun b2f(b: Boolean): Float = if (b) 1f else 0f
}
