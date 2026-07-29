package club.xiaojiawei.hsbot

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import club.xiaojiawei.hsscriptbase.config.log
import club.xiaojiawei.hsscriptcardsdk.bean.DEFAULT_WAR_SCORE_CALCULATOR
import club.xiaojiawei.hsscriptcardsdk.bean.ScoreCalculator
import club.xiaojiawei.hsscriptcardsdk.bean.War
import java.nio.FloatBuffer
import java.nio.file.Files
import java.nio.file.Path

/**
 * The ML hook. Wraps an ONNX value net as a [ScoreCalculator] (`Function<War, Double>`),
 * the exact function HS-Script's built-in MCTS calls at each search leaf. This replaces
 * the hand-tuned `WarScoreCalculatorBuilder`.
 *
 * The model consumes the 144-float vector from [WarFeatureExtractor] and outputs a single
 * value (win probability if trained with a sigmoid head). MCTS only *compares* scores, so
 * any monotonic-in-win-prob output works.
 *
 * Thread-safety: HS-Script runs MCTS across CALC_THREAD_POOL. `OrtSession.run` is safe to
 * call concurrently; we allocate a fresh input tensor per call and close it immediately.
 *
 * If the model can't be loaded, we fall back to the built-in heuristic so the bot still
 * plays (degraded) rather than crashing.
 */
class OnnxValueNet private constructor(
    private val env: OrtEnvironment,
    private val session: OrtSession,
    private val inputName: String,
) {

    fun asScoreCalculator(): ScoreCalculator = ScoreCalculator { war -> score(war) }

    private fun score(war: War): Double {
        val feats = WarFeatureExtractor.extract(war)
        // shape [1, 144]
        OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(feats),
            longArrayOf(1, WarFeatureExtractor.LENGTH.toLong()),
        ).use { input ->
            session.run(mapOf(inputName to input)).use { result ->
                val out = result[0].value
                return readScalar(out)
            }
        }
    }

    @Suppress("UNCHECKED_CAST")
    private fun readScalar(out: Any?): Double = when (out) {
        is Array<*> -> {          // [1,1] or [1]
            when (val row = out[0]) {
                is FloatArray -> row[0].toDouble()
                is Float -> row.toDouble()
                else -> error("unexpected ONNX output row: ${row?.javaClass}")
            }
        }
        is FloatArray -> out[0].toDouble()
        else -> error("unexpected ONNX output: ${out?.javaClass}")
    }

    companion object {
        /**
         * Loads the model at [modelPath]. Returns a heuristic-backed calculator on any error.
         */
        fun loadOrHeuristic(modelPath: Path): ScoreCalculator {
            return try {
                require(Files.isRegularFile(modelPath)) { "model not found: $modelPath" }
                val env = OrtEnvironment.getEnvironment()
                val session = env.createSession(modelPath.toString(), OrtSession.SessionOptions())
                val inputName = session.inputNames.first()
                log.info { "OnnxValueNet loaded: $modelPath (input=$inputName)" }
                OnnxValueNet(env, session, inputName).asScoreCalculator()
            } catch (e: Throwable) {
                log.warn(e) { "OnnxValueNet load failed, falling back to built-in heuristic" }
                DEFAULT_WAR_SCORE_CALCULATOR.build()
            }
        }
    }
}
