package com.idibon.ml.feature.indexer

import com.idibon.ml.alloy.IntentAlloy
import com.idibon.ml.feature.Feature
import com.idibon.ml.feature.tokenizer.{Tag, Token}
import java.io._
import org.apache.spark.mllib.linalg.Vectors
import org.scalatest.{Matchers, BeforeAndAfter, FunSpec}

/**
  * Tests for validating the functionality of the IndexTransformer
  *
  * @author Michelle Casbon <michelle@idibon.com>
  */

class IndexerSpec extends FunSpec with Matchers with BeforeAndAfter {

  describe("Indexer") {

    var transform: IndexTransformer = null

    before {
      transform = new IndexTransformer()
    }

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
        new Token("colorless", Tag.Word, 0, 1), new Token("green", Tag.Word, 1, 1),
        new Token("ideas", Tag.Word, 0, 1), new Token("sleep", Tag.Word, 1, 1),
        new Token("sleep", Tag.Word, 1, 1), new Token("furiously", Tag.Word, 1, 1),
        new Token("green", Tag.Word, 1, 1))
      val result1 = transform.apply(fiveTokens)

      // Save the results
      val intentAlloy = new IntentAlloy()
      transform.save(intentAlloy.writer)

      // Load the results
      val transform2 = new IndexTransformer()
      transform2.load(intentAlloy.reader, null)
      val result2 = transform2.apply(fiveTokens)

      result1 shouldBe result2
    }
  }
}
