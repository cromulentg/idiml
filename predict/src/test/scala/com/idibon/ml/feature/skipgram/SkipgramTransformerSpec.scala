package com.idibon.ml.feature.skipgram

import com.idibon.ml.feature.{StringFeature, ProductFeature}
import org.scalatest.{Matchers, FunSpec}
import com.idibon.ml.feature.tokenizer.{Tag,Token}



/**
  * Created by haley on 1/26/16.
  */
class SkipgramTransformerSpec extends FunSpec with Matchers{

  describe("apply") {
    //TODO: going to need to try some other indices
    val transform = new SkipgramTransformer(2, 2)
    val transform2 = new SkipgramTransformer(3,0)
    val transform3 = new SkipgramTransformer(2,3)
    val transform4 = new SkipgramTransformer(0,2)

    val short_tokens = Vector(Token("paper", Tag.Word, 0, 1), Token("mache", Tag.Word, 2, 5))
    val long_tokens = Vector(StringFeature("Twas"), StringFeature("brillig"),
      StringFeature(","), StringFeature("and"),StringFeature("the"),
      StringFeature("slithy"), StringFeature("toves"))

    it("should work on an empty sequence") {
      //TODO: should also return empty if k or n are longer than the sequence length
      transform(Seq[Token]()) shouldBe empty
    }

    it("should return the empty set when n=0"){
      transform2(short_tokens) shouldBe empty
    }

    it("should return the empty set when min(n,k) is larger than the input"){
      transform3(short_tokens) shouldBe empty
    }


    it("should act like ngrams when k=0"){
      transform4(short_tokens) shouldBe List(
        ProductFeature(Seq(short_tokens(0),(short_tokens(1)))))
    }

    it("should generate a correct sequence of skipgrams") {

      transform(long_tokens) shouldBe List(
        ProductFeature(List(long_tokens(0), long_tokens(1))),
        ProductFeature(List(long_tokens(0), long_tokens(2))),
        ProductFeature(List(long_tokens(0), long_tokens(3))),
        ProductFeature(List(long_tokens(1), long_tokens(2))),
        ProductFeature(List(long_tokens(1), long_tokens(3))),
        ProductFeature(List(long_tokens(1), long_tokens(4))),
        ProductFeature(List(long_tokens(2), long_tokens(3))),
        ProductFeature(List(long_tokens(2), long_tokens(4))),
        ProductFeature(List(long_tokens(2), long_tokens(5))),
        ProductFeature(List(long_tokens(3), long_tokens(4))),
        ProductFeature(List(long_tokens(3), long_tokens(5))),
        ProductFeature(List(long_tokens(3), long_tokens(6))),
        ProductFeature(List(long_tokens(4), long_tokens(5))),
        ProductFeature(List(long_tokens(4), long_tokens(6))),
        ProductFeature(List(long_tokens(5), long_tokens(6)))
      )
    }
  }
}
