package com.idibon.ml.feature.chain

import com.idibon.ml.alloy._
import com.idibon.ml.feature._
import com.idibon.ml.feature.bagofwords.Word
import com.idibon.ml.common.EmbeddedEngine

import org.json4s.JsonDSL._
import org.json4s._

import org.scalatest.{Matchers, FunSpec}

class ChainNeighborhoodSpec extends FunSpec with Matchers {

  it("should save and load correctly") {
    val archive = scala.collection.mutable.HashMap[String, Array[Byte]]()
    val xf = new ChainNeighborhood(1, 1)
    val json = xf.save(new MemoryAlloyWriter(archive))
    json shouldBe Some(JObject(List(JField("before",JInt(1)),JField("after",JInt(1)))))
    new ChainNeighborhoodLoader().load(new EmbeddedEngine, None, json) shouldBe a [ChainNeighborhood]
  }

  it("should throw an exception for overly-large values") {
    val loader = new ChainNeighborhoodLoader
    intercept[IllegalArgumentException] {
      loader.load(new EmbeddedEngine, None, Some(("before"->128) ~ ("after"->0)))
    }
    intercept[IllegalArgumentException] {
      loader.load(new EmbeddedEngine, None, Some(("before"->0) ~ ("after"->128)))
    }
  }

  it("should support empty lists") {
    val xf = new ChainNeighborhood(1, 1)
    xf(Chain()) shouldBe empty
  }

  it("should lift the entire neighborhood into chain features") {
    val features = new ChainLiftTransformer()(Chain[Feature[_]](
      "the quick brown fox".split(" ").map(Word)))
    val neighborhood = new ChainNeighborhood(2, 2)(features)
    neighborhood shouldBe Chain(
      Seq(ChainFeature(1, Word("quick")), ChainFeature(2, Word("brown"))),
      Seq(ChainFeature(-1, Word("the")), ChainFeature(1, Word("brown")), ChainFeature(2, Word("fox"))),
      Seq(ChainFeature(-1, Word("quick")), ChainFeature(-2, Word("the")), ChainFeature(1, Word("fox"))),
      Seq(ChainFeature(-1, Word("brown")), ChainFeature(-2, Word("quick"))))
  }
}
