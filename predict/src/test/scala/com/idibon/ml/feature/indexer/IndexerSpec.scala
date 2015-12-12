package com.idibon.ml.feature.indexer

import com.idibon.ml.feature.Feature
import com.idibon.ml.feature.tokenizer.{Tag, Token}
import java.io._
import org.apache.spark.mllib.linalg.Vectors
import org.scalatest.{Matchers, BeforeAndAfter, FunSpec}

class IndexerSpec extends FunSpec with Matchers with BeforeAndAfter {

  describe("Indexer") {

    var transform: IndexTransformer = null

    before {
      transform = new IndexTransformer()
    }

    it("should work on an empty sequence") {
      val empty = Seq[Feature[_]]()
      transform.apply(Map("features" -> empty)) shouldBe empty
    }

    it("should work on a sequence of Tokens") {
      val fiveTokens = Seq[Feature[Token]](
        new Token("colorless", Tag.Word, 0, 1), new Token("green", Tag.Word, 1, 1),
        new Token("ideas", Tag.Word, 0, 1), new Token("sleep", Tag.Word, 1, 1),
        new Token("furiously", Tag.Word, 1, 1))
      val expected = Seq[Feature[Index]](
        new Index(Vectors.sparse(5, Seq((0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0), (4, 1.0))))
      )
      transform.apply(Map("features" -> fiveTokens)) shouldBe expected
    }

    it("should work on a sequence of Tokens with repeats") {
      val fiveTokens = Seq[Feature[Token]](
        new Token("colorless", Tag.Word, 0, 1), new Token("green", Tag.Word, 1, 1),
        new Token("ideas", Tag.Word, 0, 1), new Token("sleep", Tag.Word, 1, 1),
        new Token("sleep", Tag.Word, 1, 1), new Token("furiously", Tag.Word, 1, 1),
        new Token("green", Tag.Word, 1, 1))
      val expected = Seq[Feature[Index]](
        new Index(Vectors.sparse(5, Seq((0, 1.0), (1, 2.0), (2, 1.0), (3, 2.0), (4, 1.0))))
      )
      transform.apply(Map("features" -> fiveTokens)) shouldBe expected
    }

    it("should fail if 'features' not provided") {
      val twoTokens = Seq[Feature[Token]](
        new Token("colorless", Tag.Word, 0, 1), new Token("green", Tag.Word, 1, 1),
        new Token("ideas", Tag.Word, 2, 1))
      intercept[java.util.NoSuchElementException] {
        transform.apply(Map("sleep" -> twoTokens))
      }
    }

    it("should save and load properly") {
      val fiveTokens = Seq[Feature[Token]](
        new Token("colorless", Tag.Word, 0, 1), new Token("green", Tag.Word, 1, 1),
        new Token("ideas", Tag.Word, 0, 1), new Token("sleep", Tag.Word, 1, 1),
        new Token("sleep", Tag.Word, 1, 1), new Token("furiously", Tag.Word, 1, 1),
        new Token("green", Tag.Word, 1, 1))
      val expected = Vectors.sparse(5, Seq((0, 1.0), (1, 2.0), (2, 1.0), (3, 2.0), (4, 1.0)))
      val result = transform.apply(Map("features" -> fiveTokens))

      // Save the results
      val filename = "/tmp/IndexerSpec.txt"
      val fos = new FileOutputStream(filename)
      val dos = new DataOutputStream(fos)
      val fis = new FileInputStream(filename)
      val dis = new DataInputStream(fis)
      for (f <- result) {
        f.save(dos)
      }

      // Load the results
      val newFeature = new Index(Vectors.dense(0.0))
      newFeature.load(dis)
      newFeature.get.values shouldBe expected

      // Remove the temporary file
      new File(filename).delete()
    }
  }
}
