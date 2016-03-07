package com.idibon.ml.feature

import com.idibon.ml.feature.bagofwords.Word
import com.idibon.ml.feature.chain.ChainFeature

import org.scalatest.{Matchers, FunSpec}

class ChainSpec extends FunSpec with Matchers {

  it("should support covariant implicit conversions") {
    val initial = Chain(0, 1, 2)
    val scaled: Chain[String] = initial.map("v = " + _.value * 2)
    scaled.flatten shouldBe List("v = 0", "v = 2", "v = 4")

    val features: Chain[ChainFeature[Feature[String]]] = scaled
      .filter(_.previous.isDefined)
      .map(_.previous.map(p => ChainFeature(-1, StringFeature(p.value))).get)

    features.flatten shouldBe List(
      ChainFeature(-1, StringFeature("v = 0")),
      ChainFeature(-1, StringFeature("v = 2"))
    )
  }

  it("should create sequences of ChainLinks") {
    val words = Seq(Word("hello"), Word("world"))
    val chain = Chain(words)
    chain.head.value shouldBe Word("hello")
    chain.head.next.get.value shouldBe Word("world")
    chain.head.next.get.next shouldBe None
    chain.head.next.get.previous.get.value shouldBe Word("hello")
    chain.head.previous shouldBe None
    chain.tail.head.value shouldBe Word("world")
    chain.tail.head.previous.get.value shouldBe Word("hello")
  }
}
