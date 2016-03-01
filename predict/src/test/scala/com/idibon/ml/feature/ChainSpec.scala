package com.idibon.ml.feature

import com.idibon.ml.feature.bagofwords.Word

import org.scalatest.{Matchers, FunSpec}

class ChainSpec extends FunSpec with Matchers {

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
