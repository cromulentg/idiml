package com.idibon.ml.predict.rules

import java.util.regex.Pattern

import com.idibon.ml.alloy.IntentAlloy
import org.apache.spark.mllib.linalg.SparseVector
import org.json4s._
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

    it("should work on an exact rule") {
      val docRules = new DocumentRules(0, List(("is", 1.0)))
      val results = docRules.getDocumentMatchCounts("this is a string")
      results.size shouldBe 1
      results.get("is") shouldBe Some(2) // 2: thIS and IS
    }

    it("should work on a regex rule") {
      val docRules = new DocumentRules(0, List(("/str[ij]ng/", 1.0)))
      val results = docRules.getDocumentMatchCounts("this is a string")
      results.size shouldBe 1
      results.get("/str[ij]ng/") shouldBe Some(1)
    }

    it("should work on two rules") {
      val docRules = new DocumentRules(0, List(("/str[ij]ng/", 1.0), ("is", 1.0)))
      val results = docRules.getDocumentMatchCounts("this is a string with is")
      results.size shouldBe 2
      results.get("/str[ij]ng/") shouldBe Some(1)
      results.get("is") shouldBe Some(3)
    }

    it("should work as expected on badly formed regex rules") {
      //missing front / so should be literal & bad regex
      val docRules = new DocumentRules(0, List(("str[ij]]ng/", 1.0), ("/*._<?>asdis/", 1.0)))
      val results = docRules.getDocumentMatchCounts("this is a string with is")
      results.size shouldBe 0 //no matches
      val results2 = docRules.getDocumentMatchCounts("/*._<?>asdis/")
      results2.size shouldBe 0 //no matches
      val results3 = docRules.getDocumentMatchCounts("str[ij]]ng/")
      results3.size shouldBe 1 //one matche
    }

    it("should work in the case a rule is not in the cache") {
      val docRules = new DocumentRules(0, List(("/str[ij]ng/", 1.0), ("is", 1.0)))
      docRules.rulesCache.remove("is") //remove is from cache.
      val results = docRules.getDocumentMatchCounts("this is a string with is")
      results.size shouldBe 1
      results.get("/str[ij]ng/") shouldBe Some(1)
      results.get("is") shouldBe None
    }
  }

  describe("save and load") {
    it("should save an empty map and load it") {
      val alloy = new IntentAlloy()
      val docRules = new DocumentRules(1, List())
      val jsonConfig = docRules.save(alloy.writer())
      //bogus stuff that should be overwritten
      val docRulesLoad = new DocumentRules(-2, List(("is", 1.0)))
      docRulesLoad.load(alloy.reader(), jsonConfig)
      docRulesLoad.rules shouldBe docRules.rules
      docRulesLoad.label shouldBe docRules.label
      //TODO: remove file
    }

    it("should save a valid map") {
      val alloy = new IntentAlloy()
      val docRules = new DocumentRules(2, List(("/str[ij]ng/", 0.3), ("is", 0.10)))
      docRules.save(new IntentAlloy().writer())
      val jsonConfig = docRules.save(alloy.writer())
      //bogus stuff that should be overwritten
      val docRulesLoad = new DocumentRules(-1, List(("is", 1.0)))
      docRulesLoad.load(alloy.reader(), jsonConfig)
      docRulesLoad.rules shouldBe docRules.rules
      docRulesLoad.label shouldBe docRules.label
      //TODO: remove file
    }
  }

  describe("populateCache") {
    it("should populate the cache correctly with the right types") {
      val docRules = new DocumentRules(0, List())
      docRules.rulesCache.size shouldBe 0
      docRules.rules = List(("/str[ij]ng/", 0.3), ("is", 0.5), ("/@#q54i7^<>(&^&?]]/", 0.5))
      docRules.populateCache()
      docRules.rulesCache.size shouldBe 3
      docRules.rulesCache.get("/str[ij]ng/").isSuccess shouldBe true
      docRules.rulesCache.get("is").isSuccess shouldBe true
      docRules.rulesCache.get("/@#q54i7^<>(&^&?]]/").isFailure shouldBe true
    }
  }

  describe("getMatches") {
    it("should return empty list on no matches") {
      val pat = Pattern.compile("is", Pattern.LITERAL | Pattern.CASE_INSENSITIVE)
      val matcher = pat.matcher("no matches whatsoever")
      (new DocumentRules(0, List())).getMatches(matcher) shouldBe List[(Int, Int)]()
    }

    it("should return a valid list with a valid match") {
      val pat = Pattern.compile("is", Pattern.LITERAL | Pattern.CASE_INSENSITIVE)
      val matcher = pat.matcher("is is is")
      new DocumentRules(0, List()).getMatches(matcher) shouldBe List[(Int, Int)]((0,2), (3,5), (6,8))
    }
  }

  describe("constructor") {
    it("Should not filter out good rule weights") {
      val docRules = new DocumentRules(2, List(("/str[ij]ng/", 1.0), ("is", 0.0), ("mon", 0.5)))
      docRules.rulesCache.size shouldBe 3
      docRules.invalidRules.size shouldBe 0
    }

    it("Should filter out bad rule weights") {
      val docRules = new DocumentRules(2, List(("/str[ij]ng/", 1.5), ("is", -0.5)))
      docRules.rulesCache.size shouldBe 0
      docRules.invalidRules.size shouldBe 2
    }
  }

  describe("docPredict") {
    it("Should return significant features") {
      val docRules = new DocumentRules(2, List(("/str[ij]ng/", 0.5), ("is", 0.5)))
      val actual = docRules.docPredict("string matching is working", true)
      actual.size shouldBe 1
      actual.getMatchCount() shouldBe 2.0
      actual.getProbability(2) shouldEqual 0.5
      actual.getSignificantFeatures(2) shouldEqual List(("/str[ij]ng/", 0.5), ("is", 0.5))
    }

    it("Should return whitelist significant feature when whitelist overrides") {
      val docRules = new DocumentRules(2, List(("/str[ij]ng/", 1.0), ("is", 0.5)))
      val actual = docRules.docPredict("string matching is working", true)
      actual.size shouldBe 1
      actual.getMatchCount() shouldBe 1.0
      actual.getProbability(2) shouldEqual 1.0
      actual.getSignificantFeatures(2) shouldEqual List(("/str[ij]ng/", 1.0))
    }

    it("Should return whitelist significant feature when blacklist overrides") {
      val docRules = new DocumentRules(2, List(("/str[ij]ng/", 0.0), ("is", 0.5)))
      val actual = docRules.docPredict("string strjng matching is working", true)
      actual.size shouldBe 1
      actual.getMatchCount() shouldBe 2.0
      actual.getProbability(2) shouldEqual 0.0
      actual.getSignificantFeatures(2) shouldEqual List(("/str[ij]ng/", 0.0))
    }

    it("Should return a document prediction result") {
      val docRules = new DocumentRules(2, List(("/str[ij]ng/", 0.5), ("is", 0.5)))
      val actual = docRules.docPredict("string matching is working", false)
      actual.size shouldBe 1
      actual.getMatchCount() shouldBe 2.0
      actual.getProbability(2) shouldEqual 0.5
      actual.getSignificantFeatures(2) shouldEqual List()
    }
  }

  describe("isRegexRule") {
    it("correctly identifies regular expression") {
      new DocumentRules(0, List()).isRegexRule("/str[ij]ng/") shouldBe true
    }

    it("correctly identifies non-regular expression") {
      new DocumentRules(0, List()).isRegexRule("/str[ij]ng") shouldBe false
      new DocumentRules(0, List()).isRegexRule("str[ij]ng") shouldBe false
      new DocumentRules(0, List()).isRegexRule(null) shouldBe false
    }
  }

  describe("calculatePseudoProbability") {
    it("should skip 0 values in count map") {
      val docRules = new DocumentRules(2, List(("/str[ij]ng/", 0.5), ("is", 0.5)))
      val countMap = Map[String,Int](("/str[ij]ng/", 0), ("is", 0))
      val (prob, count, worb) = docRules.calculatePseudoProbability(countMap)
      prob shouldBe 0.0
      count shouldBe 0.0
      worb shouldBe false
    }

    it("should return correctly with no whitelist or blacklist hit when a normal rule is hit") {
      val docRules = new DocumentRules(2, List(("/str[ij]ng/", 0.5), ("is", 0.5)))
      val countMap = Map[String,Int](("/str[ij]ng/", 1), ("is", 1))
      val (prob, count, worb) = docRules.calculatePseudoProbability(countMap)
      prob shouldBe 0.5
      count shouldBe 2.0
      worb shouldBe false
    }

    it("should return whitelist correctly") {
      val docRules = new DocumentRules(2, List(("/str[ij]ng/", 1.0), ("is", 0.5)))
      val countMap = Map[String,Int](("/str[ij]ng/", 1))
      val (prob, count, worb) = docRules.calculatePseudoProbability(countMap)
      prob shouldBe 1.0
      count shouldBe 1.0
      worb shouldBe true
    }

    it("should return blacklist correctly") {
      val docRules = new DocumentRules(2, List(("/str[ij]ng/", 0.0), ("is", 0.5)))
      val countMap = Map[String,Int](("/str[ij]ng/", 1))
      val (prob, count, worb) = docRules.calculatePseudoProbability(countMap)
      prob shouldBe 0.0
      count shouldBe 1.0
      worb shouldBe true
    }

    it("should merge blacklist and whitelist correctly if both present") {
      val docRules = new DocumentRules(2, List(("/str[ij]ng/", 0.0), ("is", 1.0)))
      val countMap = Map[String,Int](("/str[ij]ng/", 2), ("is", 8))
      val (prob, count, worb) = docRules.calculatePseudoProbability(countMap)
      prob shouldBe 0.8
      count shouldBe 10.0
      worb shouldBe true
    }

    it("should handle overriding rule hits if blacklist matched") {
      val docRules = new DocumentRules(2, List(("/str[ij]ng/", 0.0), ("is", 0.5)))
      val countMap = Map[String,Int](("/str[ij]ng/", 2), ("is", 8))
      val (prob, count, worb) = docRules.calculatePseudoProbability(countMap)
      prob shouldBe 0.0
      count shouldBe 2.0
      worb shouldBe true
    }

    it("should handle overriding rule hits if whitelist matched") {
      val docRules = new DocumentRules(2, List(("/str[ij]ng/", 1.0), ("is", 0.5)))
      val countMap = Map[String,Int](("/str[ij]ng/", 2), ("is", 8))
      val (prob, count, worb) = docRules.calculatePseudoProbability(countMap)
      prob shouldBe 1.0
      count shouldBe 2.0
      worb shouldBe true
    }

    it("should return correctly when map passed is empty") {
      val docRules = new DocumentRules(2, List(("/str[ij]ng/", 1.0), ("is", 0.5)))
      val countMap = Map[String,Int]()
      val (prob, count, worb) = docRules.calculatePseudoProbability(countMap)
      prob shouldBe 0.0
      count shouldBe 0.0
      worb shouldBe false
    }
  }

  describe("predict from vector") {
    it("Should throw exception") {
      intercept[RuntimeException] {
        new DocumentRules(0, List()).predict(new SparseVector(1, Array(0), Array(0)), false, 0.0)
      }
    }
  }

  describe("predict from JSON object") {
    it("Should predict properly when there is a match") {
      val doc = new JObject(List("content" -> new JString("this is some content")))
      val model = new DocumentRules(0, List(("/str[ij]ng/", 1.0), ("is", 1.0)))
      val actual = model.predict(doc, false, 0.0)
      actual.size shouldBe 1
      actual.getProbability(0) shouldBe 1.0
    }

    it("Should predict properly when there is no match") {
      val doc = new JObject(List("content" -> new JString("this is some content")))
      val model = new DocumentRules(0, List(("/str[ij]ng/", 1.0), ("/no[thing].*/", 1.0)))
      val actual = model.predict(doc, false, 0.0)
      actual.size shouldBe 1
      actual.getProbability(0) shouldBe 0.0
    }
  }

}
