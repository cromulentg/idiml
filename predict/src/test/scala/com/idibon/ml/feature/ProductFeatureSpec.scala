package com.idibon.ml.feature

import com.idibon.ml.feature.bagofwords.Word
import com.idibon.ml.feature.contenttype.{ContentTypeCode, ContentType}
import com.idibon.ml.feature.tokenizer.{Tag, Token}
import com.idibon.ml.feature.wordshapes.{Shape, WordShapesTransformer}
import com.idibon.ml.feature.{wordshapes => shapes}
import org.scalatest.{FunSpec, Matchers}

/**
  * Created by michelle on 2/22/16.
  */
class ProductFeatureSpec extends FunSpec with Matchers {

  it("should output human-readable strings") {
    val features = List(new StringFeature("colorless"), new StringFeature("green ideas"), new StringFeature("sleep"),
      new StringFeature("furiously"))
    val pFeature = new ProductFeature(features)

    pFeature.getHumanReadableString shouldBe Some("colorless green ideas sleep furiously")
  }

  it("should return None if any feature is None") {
    // Must cast List to the Feature supertype for this to work
    val featureList = List[Feature[_]](new StringFeature("colorless"), new StringFeature("green"),
      new StringFeature("ideas"), Shape("Cc"), new StringFeature("furiously"))
    val pFeature = new ProductFeature(featureList)

    pFeature.getHumanReadableString shouldBe None
  }
}
