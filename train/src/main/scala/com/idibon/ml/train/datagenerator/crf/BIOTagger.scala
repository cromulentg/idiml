package com.idibon.ml.train.datagenerator.crf

import com.idibon.ml.feature.tokenizer.Token
import org.json4s._
import com.idibon.ml.predict.crf._
import com.idibon.ml.feature.{Chain, ChainPipeline, SequenceGenerator}
import com.idibon.ml.train.datagenerator.json._
import org.apache.spark.mllib.linalg.Vector

/** Converts documents with annotations into sequence of tagged feature vectors
  *
  */
trait BIOTagger {

  /** Converts documents into a sequence of non-overlapping tokens */
  def sequenceGenerator: SequenceGenerator

  /** Converts token sequences into feature vector sequences */
  def featureExtractor: ChainPipeline

  /** Converts annotated documents into BIO-tagged training data
    *
    * Applies sequenceGenerator and featureExtractor to the document to
    * create a sequence of feature vectors, and tags each entry in the
    * sequence with a BIOTag based on the document's annotations
    *
    * Annotations must be non-overlapping and include offsets and lengths.
    *
    * @param json the document to process
    * @return tagged sequence
    */
  def tag(json: JObject): Iterable[(BIOTag, Vector)] = {
    implicit val formats = org.json4s.DefaultFormats

    val doc = json.extract[Document]
    require(doc.annotations.forall(_.isSpan), "Detected non-span annotation")

    val tokens = sequenceGenerator(json)
    val tags: Traversable[(Token, BIOTag, Option[Annotation])] = getTokenTags(tokens, doc)

    // map the tokens to feature vectors, combine with tags and return
    val features = featureExtractor(json, tokens).map(_.value)
    tags.map(_._2).toIterable.zip(features.toIterable)
  }

  /**
    * Gets tokens & their tags from a chain of tokens and a document with gold annotations.
    *
    * @param tokens
    * @param doc
    * @return
    */
  def getTokenTags(tokens: Chain[Token],
                   doc: Document): Traversable[(Token, BIOTag, Option[Annotation])] = {
    var anns = getAnnotations(doc)
    // track the most-recent span annotation begun from the annotations list
    var curr: Annotation = null
    /* generate tags for each of the tokens in the sequence by comparing
     * the token locations against the annotation location. the first token
     * within each annotation will be tagged (BEGIN, <label.name>), subsequent
     * tokens will be tagged (INSIDE, <label.name>). */
    val tags = tokens.map(tok => {
      val t = tok.value
      if (curr != null && t.offset < curr.offset.get + curr.length.get) {
        /* already inside a span and at least part of this token overlaps
       * the annotation, so create an INSIDE tag */
        (t, BIOLabel(BIOType.INSIDE, curr.label.name), Some(curr))
      } else if (anns.isEmpty) {
        /* token is beyond the current span's limits and no more annotations
       * exist, so generate an OUTSIDE tag */
        (t, BIOOutside, None)
      } else {
        curr = null
        /* advance the annotations list past any annotations that were wholly
       * contained within the previous token, due to a severe misalignment
       * between the generated tokens and the spans. this might also happen
       * if, for example, a span was defined within a whitespace region that
       * was discarded by sequenceGenerator. */
        while (!anns.isEmpty && anns.head.end.get <= t.offset)
          anns = anns.tail

        /* if this token starts somewhere within the head annotation, begin a
       * new span; otherwise, this token is outside all spans */
        if (!anns.isEmpty && anns.head.contains(t)) {
          curr = anns.head
          anns = anns.tail
          (t, BIOLabel(BIOType.BEGIN, curr.label.name), Some(curr))
        } else {
          (t, BIOOutside, None)
        }
      }
    })
    tags
  }

  /**
    * Grabs annotations that are positive and have a positive length.
    * @param doc
    * @return
    */
  def getAnnotations(doc: Document): List[Annotation] = {
    /* discard negative and zero-length annotations (any untagged
     * token will be marked as OUTSIDE, and sort the remaining
     * annotations by increasing offset, to match the token order */
    val anns = doc.annotations
      .filter(ann => ann.isPositive && ann.length.get > 0)
      .sortBy(_.offset.get)
    assert(!BIOTagger.overlaps(anns), "Overlapping annotations")
    anns
  }
}

private[crf] object BIOTagger {

  /** Returns true if any annotations in the list overlap
    *
    * Annotations should be sorted by increasing offset before calling this
    * method, and all Annotations must be spans.
    *
    * @param annotations list of annotations
    * @return true if any annotations overlap
    */
  def overlaps(annotations: Seq[Annotation]): Boolean = {
    annotations.sliding(2).exists(_ match {
      case a :: b :: nil => a.end.get > b.offset.get
      case _ => false
    })
  }
}
