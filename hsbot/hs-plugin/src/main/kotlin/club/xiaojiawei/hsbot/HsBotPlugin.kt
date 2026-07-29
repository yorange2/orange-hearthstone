package club.xiaojiawei.hsbot

import club.xiaojiawei.hsscriptstrategysdk.StrategyPlugin

/**
 * Plugin entry point, discovered by HS-Script via META-INF/services.
 */
class HsBotPlugin : StrategyPlugin {
    override fun description(): String = "ONNX value-net MCTS strategy (SabberStone-trained)."
    override fun author(): String = "orange-hearthstone"
    override fun version(): String = VersionInfo.VERSION
    override fun id(): String = "orange-hearthstone.hs-bot"
    override fun name(): String = "HS Bot (ValueNet)"
    override fun homeUrl(): String = "https://github.com/orange-hearthstone/hs-bot"

    override fun cardSDKVersion(): String? =
        if (VersionInfo.CARD_SDK_VERSION_USED.endsWith("}")) null else VersionInfo.CARD_SDK_VERSION_USED

    override fun strategySDKVersion(): String? =
        if (VersionInfo.STRATEGY_SDK_VERSION_USED.endsWith("}")) null else VersionInfo.STRATEGY_SDK_VERSION_USED
}
