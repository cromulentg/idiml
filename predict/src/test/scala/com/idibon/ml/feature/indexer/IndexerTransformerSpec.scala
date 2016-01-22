package com.idibon.ml.feature.indexer

import com.idibon.ml.alloy.IntentAlloy
import com.idibon.ml.feature.Feature
import com.idibon.ml.feature.tokenizer.{Tag, Token}
import com.idibon.ml.common.EmbeddedEngine
import org.apache.spark.mllib.linalg.Vectors
import org.scalatest.{Matchers, BeforeAndAfter, FunSpec}

/**
  * Tests for validating the functionality of the IndexTransformer
  *
  * @author Michelle Casbon <michelle@idibon.com>
  */

class IndexerTransformerSpec extends FunSpec with Matchers with BeforeAndAfter {

  var transform: IndexTransformer = null

  before {
    transform = new IndexTransformer()
  }

  describe("Indexer") {

    it("should work on an empty sequence") {
      val emptyFeatures = Seq[Feature[_]]()
      val emptyVector = Vectors.zeros(0)
      transform.apply(emptyFeatures) shouldBe emptyVector
    }

    it("should work on a sequence of Tokens") {
      val fiveTokens = Seq[Feature[Token]](
        new Token("colorless", Tag.Word, 0, 1), new Token("green", Tag.Word, 1, 1),
        new Token("ideas", Tag.Word, 0, 1), new Token("sleep", Tag.Word, 1, 1),
        new Token("furiously", Tag.Word, 1, 1))
      val expected = Vectors.sparse(5, Seq((0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0), (4, 1.0)))
      transform.apply(fiveTokens) shouldBe expected
    }

    it("should work on a sequence of Tokens with repeats") {
      val fiveTokens = Seq[Feature[Token]](
        new Token("colorless", Tag.Word, 0, 1), new Token("green", Tag.Word, 1, 1),
        new Token("ideas", Tag.Word, 0, 1), new Token("sleep", Tag.Word, 1, 1),
        new Token("sleep", Tag.Word, 1, 1), new Token("furiously", Tag.Word, 1, 1),
        new Token("green", Tag.Word, 1, 1))
      val expected = Vectors.sparse(5, Seq((0, 1.0), (1, 2.0), (2, 1.0), (3, 2.0), (4, 1.0)))
      transform.apply(fiveTokens) shouldBe expected
    }

    it("should save and load a transformer properly") {
      val fiveTokens = Seq[Feature[Token]](
        new Token("colorless", Tag.Word, 0, 0), new Token("green", Tag.Word, 0, 0),
        new Token("ideas", Tag.Word, 0, 0), new Token("sleep", Tag.Word, 0, 0),
        new Token("sleep", Tag.Word, 0, 0), new Token("furiously", Tag.Word, 0, 0),
        new Token("green", Tag.Word, 0, 0))
      transform.apply(fiveTokens)

      // Save the results
      val intentAlloy = new IntentAlloy()
      transform.save(intentAlloy.writer)

      // Load the results
      val transform2 = (new IndexTransformLoader).load(new EmbeddedEngine, intentAlloy.reader, null)

      transform.getFeatureIndex shouldBe transform2.getFeatureIndex
    }

    it("should give the same result after applying twice") {
      val fiveTokens = Seq[Feature[Token]](
        new Token("colorless", Tag.Word, 0, 1), new Token("green", Tag.Word, 1, 1),
        new Token("ideas", Tag.Word, 0, 1), new Token("sleep", Tag.Word, 1, 1),
        new Token("furiously", Tag.Word, 1, 1))
      val expected = Vectors.sparse(5, Seq((0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0), (4, 1.0)))
      transform.apply(fiveTokens) shouldBe expected
      transform.apply(fiveTokens) shouldBe expected
    }
  }

  describe("Prune features tests") {

    def predicte1(num:Int): Boolean = {
      !List(0, 1, 2, 3, 4).contains(num)
    }
    def predicte2(num:Int): Boolean = {
      !List(10, 20, 30, 40, 4).contains(num)
    }
    it("should work on empty index") {
      transform.prune(predicte1)
      transform.getFeatureIndex.isEmpty shouldBe true
    }
    it("should work on non-empty index where they are used and thus should not be removed") {
      val fiveTokens = Seq[Feature[Token]](
        new Token("colorless", Tag.Word, 0, 1), new Token("green", Tag.Word, 1, 1),
        new Token("ideas", Tag.Word, 0, 1), new Token("sleep", Tag.Word, 1, 1),
        new Token("furiously", Tag.Word, 1, 1))
      transform.apply(fiveTokens)
      transform.prune(predicte1)
      transform.getFeatureIndex.size shouldBe 5
    }
    it("should work on non-empty index where some indexes are not used and thus should be removed") {
      val fiveTokens = Seq[Feature[Token]](
        new Token("colorless", Tag.Word, 0, 1), new Token("green", Tag.Word, 1, 1),
        new Token("ideas", Tag.Word, 0, 1), new Token("sleep", Tag.Word, 1, 1),
        new Token("furiously", Tag.Word, 1, 1))
      transform.apply(fiveTokens)
      transform.prune(predicte2)
      transform.getFeatureIndex.size shouldBe 1
      transform.getFeatureIndex.getOrElse(fiveTokens(4), 0) shouldBe 4
    }
  }
}
