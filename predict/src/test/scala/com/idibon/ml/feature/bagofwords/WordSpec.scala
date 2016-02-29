package com.idibon.ml.feature.bagofwords

import org.scalatest.{BeforeAndAfter, Matchers, FunSpec}

class WordSpec extends FunSpec with Matchers with BeforeAndAfter {

  describe("get") {
    it("Word.get should output the full feature") {
      val wordFeatureHello = new Word("Hello")
      val wordFeatureWorld = new Word("world!")

      wordFeatureHello.get shouldBe Word("Hello")
      wordFeatureWorld.get shouldBe Word("world!")
    }

    it("Word.getAsString should output human-readable strings") {
      val wordFeatureHello = new Word("Hello")
      val wordFeatureWorld = new Word("world!")

      wordFeatureHello.getAsString shouldBe Some("Hello")
      wordFeatureWorld.getAsString shouldBe Some("world!")
    }
  }
}
