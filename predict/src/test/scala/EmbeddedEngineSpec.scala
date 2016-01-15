package com.idibon.ml.predict

import org.scalatest._
import com.idibon.ml.common.Engine

class EmbeddedEngineSpec extends FunSpec with Matchers {

  describe("EmbeddedEngine") {

    it("should call the constructor correctly") {
      val engine = new EmbeddedEngine()
      engine shouldBe an[Engine]
    }

  }
}
