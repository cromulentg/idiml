package com.idibon.ml.predict.rules

import org.scalatest.{Matchers, BeforeAndAfter, FunSpec}

/**
  * Class to test the Document Rules Model.
  */
class DocumentRulesSpec extends FunSpec with Matchers with BeforeAndAfter {

  describe("getDocumentMatchCounts") {

    before {
    }

    it("should work on no rules") {
      val docRules = new DocumentRules(0, List())
      val results = docRules.getDocumentMatchCounts("this is a string")
      results.size shouldBe 0
    }

    it("should work on exact rule") {
      val docRules = new DocumentRules(0, List(("is", 1.0)))
      val results = docRules.getDocumentMatchCounts("this is a string")
      results.size shouldBe 1
      results.get("is") shouldBe Some(2) // 2: thIS and IS
    }

    it("should work on reg ex rule") {
      val docRules = new DocumentRules(0, List(("/str[ij]ng/", 1.0)))
      val results = docRules.getDocumentMatchCounts("this is a string")
      results.size shouldBe 1
      results.get("/str[ij]ng/") shouldBe Some(1)
    }

    it("should work on reg two regex rules") {
      val docRules = new DocumentRules(0, List(("/str[ij]ng/", 1.0), ("is", 1.0)))
      val results = docRules.getDocumentMatchCounts("this is a string with is")
      results.size shouldBe 2
      results.get("/str[ij]ng/") shouldBe Some(1)
      results.get("is") shouldBe Some(3)
    }
  }

  describe("docPredict") {
    //TODO:
  }

  describe("createRegexExpression") {
    //TODO:
  }

  describe("isRegexRule") {
    //TODO:
  }

  describe("calculatePseudoProbability") {
    //TODO:
  }

  describe("predict") {
    //TODO:
  }

}
