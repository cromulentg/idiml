package com.idibon.ml.feature

import org.scalatest.{Matchers, FunSpec}

/**
  * Created by michelle on 2/22/16.
  */
class FeatureSpec extends FunSpec with Matchers {

  it("should output human-readable strings") {
    val sFeatureHello = new StringFeature("hello")
    val sFeatureWorld = new StringFeature("world")

    sFeatureHello.get shouldBe "hello"
    sFeatureWorld.get shouldBe "world"
    sFeatureHello.getAsString shouldBe Some("hello")
    sFeatureWorld.getAsString shouldBe Some("world")
  }
}
