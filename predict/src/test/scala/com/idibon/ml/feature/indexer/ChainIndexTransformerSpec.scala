package com.idibon.ml.feature.indexer

import org.scalatest.{Matchers, FunSpec}

import com.idibon.ml.feature._

import org.apache.spark.mllib.linalg.Vectors

class ChainIndexTransformerSpec extends FunSpec with Matchers {

  it("should add features to the vocabulary as they are observed") {
    val x = new ChainIndexTransformer(new MutableVocabulary())
    val chain = Chain(
      Seq[Feature[_]](StringFeature("a"), StringFeature("b")),
      Seq[Feature[_]](bagofwords.Word("a"), bagofwords.Word("b")),
      Seq[Feature[_]](StringFeature("a"), bagofwords.Word("a"), StringFeature("a")))
    x(chain) shouldBe Chain(
      Vectors.dense(1.0, 1.0),
      Vectors.dense(0.0, 0.0, 1.0, 1.0),
      Vectors.dense(2.0, 0.0, 1.0, 0.0))
  }

  it("should return empty vectors for OOV features") {
    val thawed = new ChainIndexTransformer(new MutableVocabulary())
    thawed(Chain(Seq[Feature[_]](bagofwords.Word("a"))))
    val frozen = thawed.freeze
    val chain = Chain(
      Seq[Feature[_]](StringFeature("a"), StringFeature("b")),
      Seq[Feature[_]](bagofwords.Word("c")))
    frozen(chain) shouldBe Chain(Vectors.dense(0.0), Vectors.dense(0.0))
  }

  it("should work with empty chains") {
    val thawed = new ChainIndexTransformer(new MutableVocabulary())
    thawed(Chain(Seq[Feature[_]](bagofwords.Word("a"))))
    val frozen = thawed.freeze
    frozen(Chain(Seq())) shouldBe Chain(Seq())
    frozen(Chain(Seq[Feature[_]](StringFeature("a")))) shouldBe Chain(Vectors.zeros(1))
  }

  it("should work with empty vocabularies") {
    val x = new ChainIndexTransformer(new ImmutableVocabulary(Map(), 0))
    x(Chain(Seq[Feature[_]](StringFeature("a")))) shouldBe Chain(Vectors.zeros(0))
    val y = new ChainIndexTransformer(new ImmutableVocabulary(Map(), 100))
    y(Chain(Seq[Feature[_]](StringFeature("a")))) shouldBe Chain(Vectors.zeros(100))
  }

}
