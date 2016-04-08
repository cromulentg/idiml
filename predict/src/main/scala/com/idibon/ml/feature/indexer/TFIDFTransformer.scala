package com.idibon.ml.feature.indexer

import com.idibon.ml.alloy.Alloy.{Writer, Reader}
import com.idibon.ml.common.{Engine, Archivable}
import com.idibon.ml.feature.{FeatureOutputStream, FeatureInputStream, Feature, Freezable}
import com.typesafe.scalalogging.StrictLogging
import org.apache.spark.mllib.linalg.{SparseVector, Vectors, Vector}
import org.json4s.JsonAST.JField
import org.json4s.{JInt, JObject}


/**
  * TFIDF Transformer. This is used in place of IndexTransformer.
  *
  * Rather than outputting the term frequency (TF) of a feature in relation
  * to a document, it outputs the term frequency inverse document frequency (TF-IDF) value.
  *
  * Uses the vocabulary class underneath to control changing features into dimensions,
  * and delegates to the IDFCalculator class to keep track of document frequencies and values
  * required to compute and IDF value for a given feature dimension.
  *
  * @author "Stefan Krawczyk <stefan@idibon.com>" on 4/7/16.
  * @param vocabulary maps feature -> dimension index
  * @param iDFCalculator maps index -> IDF value
  */
class TFIDFTransformer(private[indexer] val vocabulary: Vocabulary,
                       private[indexer] val iDFCalculator: IDFCalculator)
  extends AbstractIndexTransformer(vocabulary)
    with Archivable[TFIDFTransformer, TFIDFTransformerLoader]
    with Freezable[TFIDFTransformer] with StrictLogging {

  def this() { this(new MutableVocabulary(), new MutableIDFCalculator()) }

  def apply(features: Seq[Feature[_]]*): Vector = toVector(features.flatten)

  def freeze(): TFIDFTransformer = {
    new TFIDFTransformer(vocabulary.freeze, iDFCalculator.freeze)
  }

  /**
    * Overrides the parent toVector implementation to also perform the IDF calculation.
    *
    * @param features list of features
    * @return vector of feature counts
    */
  protected override def toVector(features: Seq[Feature[_]]): SparseVector = {
    val sparseVect = super.toVector(features)
    iDFCalculator.incrementTotalDocCount()
    iDFCalculator.incrementSeenCount(sparseVect.indices)
    val idfs = iDFCalculator.inverseDocumentFrequency(sparseVect.indices)
    val tfidfs = sparseVect.values.zip(idfs).map({case (tf, idf) => tf * idf})
    Vectors.sparse(sparseVect.size, sparseVect.indices, tfidfs).asInstanceOf[SparseVector]
  }

  /** Removes Features from the vocabulary using a predicate function
    *
    * Removes any feature from the vocabulary where the provided predicate
    * function returns true when called using the feature's dimension.
    *
    * @param pred predicate function returning true for pruned features
    */
  override def prune(pred: (Int) => Boolean) = {
    super.prune(pred)
    iDFCalculator.prune(pred)
  }

  /** Saves the transform state to an alloy
    *
    * @param writer alloy output interface
    * @return configuration JSON
    */
  override def save(writer: Writer): Option[JObject] = {
    val Some(config) = super.save(writer)
    val fos = new FeatureOutputStream(
      writer.resource(AbstractIndexTransformer.IDF_RESOURCE_NAME))
    try {
      iDFCalculator.save(fos)
    } finally {
      fos.close()
    }
    Some(JObject(List(
      JField("minimumObservations", (config \ "minimumObservations").asInstanceOf[JInt]),
      JField("minimumDocumentObservations", JInt(iDFCalculator.minimumDocumentObservations))
    )))
  }
}

/** Paired loader class for IndexTransformer */
class TFIDFTransformerLoader extends AbstractIndexTransformLoader[TFIDFTransformer] {
  /** Not implemented since we also need an IDFCalculator **/
  protected def newTransform(v: Vocabulary) = throw new NotImplementedError()

  /** Method that invokes creating a TFIDFTransfomer from a vocab & idf calculator. */
  protected def newTransform(v: Vocabulary, i: IDFCalculator) = new TFIDFTransformer(v, i)

  /**
    * Overrides the parent method to handle also loading the IDF Calculator.
    *
    * @param engine implementation of the Engine trait
    * @param reader location within Alloy for loading any resources
    *   previous preserved by a call to save
    * @param config archived configuration data returned by a previous call to save
    * @return this object
    */
  override def load(engine: Engine,
                    reader: Option[Reader],
                    config: Option[JObject]): TFIDFTransformer = {
    val vocab = loadVocabulary(reader, config)
    val idfCalc = loadIDFCalculator(reader, config)
    newTransform(vocab, idfCalc)
  }

  /**
    * Loads the IDF Calculator if there is one to load, else returns a mutable IDF Calculator.
    *
    * @param reader location within Alloy for loading any resources
    *   previous preserved by a call to save
    * @param config archived configuration data returned by a previous
    * @return a IDFCalculator that maps dimension index -> IDF value
    */
  def loadIDFCalculator(reader: Option[Reader],
              config: Option[JObject]): IDFCalculator = {
    implicit val formats = org.json4s.DefaultFormats
    val observations = config.map(_ \ "minimumDocumentObservations")
      .collect({case j: JInt => j})
      .map(_.num.intValue())
      .getOrElse(1)
    val idfCalc = reader match {
      case None => new MutableIDFCalculator
      case Some(reader) => {
        val fis = new FeatureInputStream(
          reader.resource(AbstractIndexTransformer.IDF_RESOURCE_NAME))
        try {
          IDFCalculator.load(fis)
        } finally {
          fis.close()
        }
      }
    }
    idfCalc.minimumDocumentObservations = observations
    idfCalc
  }
}
