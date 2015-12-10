import org.json4s._
import org.apache.spark.mllib.linalg.Vector
import com.idibon.ml.alloy.Alloy

package com.idibon.ml.feature {

  /** The feature pipeline transforms documents into feature vectors
    * 
    * 
    */
  trait FeaturePipeline extends Archivable {
    val spaces: Seq[FeatureSpace[_]]

    /** Applies the entire feature pipeline to the provided document.
      * 
      * Returns a sequence of Vectors, one for each FeatureSpace included
      * in this pipeline.
      */
    def apply(document: Any): Seq[Vector] = {
      List[Vector]()
    }
  }

  /** Example configuration:
    *  "spaces": [
    *    {
    *      "name": "word_vectors",
    *      "class": "Word2VecSpace",
    *      "input": {
    *        "words": "tokens"
    *      }
    *    },
    *    {
    *      "name": "indexed_word_shapes",
    *      "class": "IndexSpace",
    *      "input": {
    *        "feature": "word_shapes"
    *      }
    *    },
    *    {
    *      "name": "indexed_tlds",
    *      "class": "IndexSpace",
    *      "input": {
    *        "feature": "metadata_url_tld"
    *      }
    *    }
    *  ],
    *  "transformations": [
    *    {
    *      "name": "tokens",
    *      "class": "Tokenizer",
    *      "input": {
    *        "content": "$content"  (reserved reference to special document content feature)
    *      }
    *    },
    *    {
    *      "name": "word_shapes",
    *      "class": "WordShapeTransformer",
    *      "input": {
    *        "words": "tokens"
    *      },
    *      "params": {
    *        "ignore_punctuation": true
    *      }
    *    }
    *  ]
    */
}
