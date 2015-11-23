package com.idibon.ml.predict

import org.scalatest._

class EmbeddedEngineSpec extends FunSpec with Matchers {

  describe("EmbeddedEngine") {

    it("should call the constructor correctly") {
      val engine = new EmbeddedEngine
      engine shouldBe an[Engine]
    }

  }
}