package club.xiaojiawei.hsbot

import club.xiaojiawei.hsscriptcardsdk.bean.Card
import club.xiaojiawei.hsscriptcardsdk.bean.Player
import club.xiaojiawei.hsscriptcardsdk.bean.TEST_CARD_ACTION
import club.xiaojiawei.hsscriptcardsdk.bean.War
import club.xiaojiawei.hsscriptcardsdk.enums.CardTypeEnum

/**
 * Builds an HS-Script [War] from the engine-neutral fixture `state` (see
 * hsbot/fixtures/README.md), so [WarFeatureExtractor] can be checked against the reference
 * vectors. Cards use the no-op [TEST_CARD_ACTION]; the extractor never invokes actions.
 *
 * Field ↔ engine mapping notes (must match FEATURES.md):
 *  - hero effective HP = blood() = (health + armor) - damage; we set damage=0 and
 *    health = effectiveHp - armor, armor = armor, so blood()=effectiveHp and
 *    remaining armor = max(armor - damage, 0) = armor.
 *  - minion remaining health = blood() = health (armor/damage 0).
 *  - weapon remaining durability = blood() = durability.
 *  - canAttack is forced via isExhausted = !canAttack (attack>0 assumed when canAttack).
 */
object TestWarBuilder {

    @Suppress("UNCHECKED_CAST")
    fun build(state: Map<String, Any?>): War {
        val war = War(false)
        val me = Player(allowLog = false, playerId = "1", gameId = "me", war = war)
        val opp = Player(allowLog = false, playerId = "2", gameId = "opp", war = war)
        war.me = me
        war.rival = opp
        war.player1 = me
        war.player2 = opp
        war.warTurn = num(state["turn"])
        war.firstPlayerGameId = if (state["meIsFirst"] == true) me.gameId else opp.gameId

        fill(me, state["me"] as Map<String, Any?>)
        fill(opp, state["opp"] as Map<String, Any?>)
        return war
    }

    @Suppress("UNCHECKED_CAST")
    private fun fill(p: Player, s: Map<String, Any?>) {
        val hero = s["hero"] as Map<String, Any?>
        val effHp = num(hero["effectiveHp"])
        val armor = num(hero["armor"])
        p.playArea.hero = card(CardTypeEnum.HERO).apply {
            this.health = effHp - armor
            this.armor = armor
        }

        (s["weapon"] as? Map<String, Any?>)?.let { w ->
            p.playArea.weapon = card(CardTypeEnum.WEAPON).apply {
                this.atc = num(w["attack"])
                this.durability = num(w["durability"])
            }
        }

        for (m in (s["board"] as List<Map<String, Any?>>)) {
            p.playArea.cards.add(card(CardTypeEnum.MINION).apply {
                atc = num(m["attack"])
                health = num(m["health"])
                isExhausted = !(m["canAttack"] as Boolean)
                isTaunt = m["taunt"] as Boolean
                isDivineShield = m["divineShield"] as Boolean
                isWindFury = m["windfury"] as Boolean
                isLifesteal = m["lifesteal"] as Boolean
                isPoisonous = m["poisonous"] as Boolean
            })
        }

        repeat(num(s["hand"])) { p.handArea.cards.add(card(CardTypeEnum.MINION)) }
        repeat(num(s["deck"])) { p.deckArea.cards.add(card(CardTypeEnum.MINION)) }
        repeat(num(s["secrets"])) { p.secretArea.cards.add(card(CardTypeEnum.SPELL)) }

        val mana = s["mana"] as Map<String, Any?>
        p.resources = num(mana["max"])
        p.overloadLocked = num(mana["overload"])
        p.fatigue = num(s["fatigue"])
    }

    private var entitySeq = 0
    private fun card(type: CardTypeEnum): Card =
        Card(TEST_CARD_ACTION).apply {
            cardType = type
            entityId = "e${entitySeq++}"
        }

    private fun num(v: Any?): Int = (v as Number).toInt()
}
