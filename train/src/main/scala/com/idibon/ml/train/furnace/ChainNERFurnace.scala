package com.idibon.ml.train.furnace

import scala.util.Random

import com.idibon.ml.predict._
import com.idibon.ml.feature._
import com.idibon.ml.common.Engine
import com.idibon.ml.train.TrainOptions
import com.idibon.ml.train.datagenerator.crf._

import org.json4s._

/** Furnace for generating Chain NER models using ConLL-style BIO tags
  *
  * @param name Name of the furnace (and the generated model within the Alloy)
  * @param sequenceGenerator sequence generator, must not be frozen
  * @param featureExtractor feature extractor, must not be frozen
  * @param prng random number generator instance to use
  */
class ChainNERFurnace(val name: String,
  baseSequenceGenerator: SequenceGenerator,
  baseFeatureExtractor: ChainPipeline,
  private[furnace] val prng: Random) extends Furnace2[Span] with BIOTagger {

  private[this] var _currentSequenceGenerator = baseSequenceGenerator
  private[this] var _currentFeatureExtractor = baseFeatureExtractor

  def featureExtractor = _currentFeatureExtractor
  def sequenceGenerator = _currentSequenceGenerator

  /** Trains the model
    *
    * @param options training options and data
    * @return the trained model
    */
  protected def doTrain(options: TrainOptions): PredictModel[Span] = {
    /* prime the feature pipelines, and return the total number of
     * dimensions in the feature space (size of the last vector in the
     * last extant chain) */
    val dimensions = options.documents().foldLeft(0)((dims, doc) => {
      /* if the last document(s) all convert to empty sequences, return
       * the previous dimension value */
      featureExtractor(doc, sequenceGenerator(doc))
        .lastOption
        .map(_.value.size)
        .getOrElse(dims)
    })
    // freeze the now-primed pipelines
    _currentSequenceGenerator = sequenceGenerator.freeze()
    _currentFeatureExtractor = featureExtractor.freeze()

    // TODO: figure out how to pass regularization parameters...
    val model = new crf.FactorieCRF(dimensions) with crf.TrainableFactorieModel

    /* generate training data by tagging all of the documents with the
     * appropriate B/I/O sequence, and observing everything in the model */
    val trainingData = options.documents().map(json => {
      model.observe(tag(json).map({ case (t, v) => (t.toString, v) }))
    }).toSeq

    // make sure that lazy training data views are all observed pre-training
    trainingData.foreach(x => x)

    model.train(trainingData, prng)

    new crf.BIOModel(model, sequenceGenerator, featureExtractor)
  }
}

object ChainNERFurnace extends Function3[Engine, String, JObject, Furnace2[Span]] {

  /** Constructs a ChainNERFurnace to train NER models
    *
    * @param name furnace / model name
    * @param engine current IdiML engine content
    * @param json JSON configuration data for the CRFFurnace
    */
  def apply(engine: Engine, name: String, json: JObject): ChainNERFurnace = {
    implicit val formats = org.json4s.DefaultFormats
    val config = json.extract[ChainNERFurnaceConfig]

    val prng = new Random(config.seed.getOrElse(
      java.lang.Float.floatToIntBits(math.Pi.toFloat)))

    val sequenceGenerator = (new SequenceGeneratorLoader).load(engine,
      None, Some(config.sequenceGenerator))

    val featureExtractor = (new ChainPipelineLoader).load(engine,
      None, Some(config.featureExtractor))

    new ChainNERFurnace(name, sequenceGenerator, featureExtractor, prng)
  }
}

/** Configuration schema for the CRFFurnace
  *
  * @param sequenceGenerator configuration of model sequence generator
  * @param featureExtractor configuration of model feature extractor
  * @param seed optional seed value for random number generator
  */
case class ChainNERFurnaceConfig(sequenceGenerator: JObject,
  featureExtractor: JObject,
  seed: Option[Int])
