package club.xiaojiawei.hsbot

import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.DynamicTest
import org.junit.jupiter.api.TestFactory
import java.nio.file.Files
import java.nio.file.Path
import kotlin.streams.toList

/**
 * Parity guard: the Kotlin [WarFeatureExtractor] must reproduce every fixture's `expected`
 * vector within 1e-6. The same fixtures gate the C# extractor, keeping the two in lockstep.
 *
 * Fixtures live at repo `hsbot/fixtures/*.json`; resolved relative to this module.
 */
class FeatureParityTest {

    private val mapper = jacksonObjectMapper()

    @TestFactory
    fun `extractor matches every fixture`(): List<DynamicTest> {
        val dir = fixturesDir()
        val files = Files.list(dir).use { s ->
            s.filter { it.toString().endsWith(".json") }.toList().sortedBy { it.fileName.toString() }
        }
        check(files.isNotEmpty()) { "no fixtures found in $dir" }

        return files.map { file ->
            DynamicTest.dynamicTest(file.fileName.toString()) {
                @Suppress("UNCHECKED_CAST")
                val doc = mapper.readValue(file.toFile(), Map::class.java) as Map<String, Any?>
                assertEquals("v1", doc["version"], "fixture ${file.fileName} is not contract v1")

                val state = doc["state"] as Map<String, Any?>
                val expected = (doc["expected"] as List<Number>).map { it.toFloat() }

                val war = TestWarBuilder.build(state)
                val actual = WarFeatureExtractor.extract(war)

                assertEquals(expected.size, actual.size, "vector length")
                for (i in expected.indices) {
                    assertEquals(
                        expected[i], actual[i], 1e-6f,
                        "feature[$i] mismatch in ${file.fileName}",
                    )
                }
            }
        }
    }

    private fun fixturesDir(): Path {
        // module dir = hsbot/hs-plugin ; fixtures = ../fixtures
        val candidates = listOf(
            Path.of("..", "fixtures"),
            Path.of("hsbot", "fixtures"),
            Path.of(System.getProperty("user.dir")).resolve("../fixtures"),
        )
        return candidates.firstOrNull { Files.isDirectory(it) }
            ?: error("could not locate hsbot/fixtures from ${System.getProperty("user.dir")}")
    }
}
