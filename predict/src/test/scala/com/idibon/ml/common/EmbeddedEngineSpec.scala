package com.idibon.ml.common

import org.scalatest.{Matchers, FunSpec}

class EmbeddedEngineSpec extends FunSpec with Matchers {

  describe("encodeUsername") {
    it("should create 48-bit hexstrings") {
      EmbeddedEngine.encodeUsername("") shouldBe "da39a3ee5e6b"
      EmbeddedEngine.encodeUsername("Foo/Bar") shouldBe "e3d258d15bbd"
    }
  }
}
