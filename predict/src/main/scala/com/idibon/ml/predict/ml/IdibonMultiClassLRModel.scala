package com.idibon.ml.predict.ml

import java.io.{IOException, DataInputStream, DataOutputStream}

import com.idibon.ml.alloy.Alloy.{Writer, Reader}
import com.idibon.ml.alloy.Codec
import com.idibon.ml.common.{Archivable, ArchiveLoader, Engine}
import com.idibon.ml.feature.{FeaturePipelineLoader, FeaturePipeline}
import com.idibon.ml.predict.{MultiLabelDocumentResultBuilder, PredictResult, PredictOptions}
import com.typesafe.scalalogging.StrictLogging
import org.apache.spark.mllib.classification.IdibonSparkMLLIBLRWrapper
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.json4s._


/**
  * Class that wraps our extended Spark MLLIB Multinomial LR model.
  *
  * This means that this class returns a MultiLabelDocumentResult and
  * it should be used with a GangModel, rather than an EnsembleModel.
  *
  * @author "Stefan Krawczyk <stefan@idibon.com>"
  */
case class IdibonMultiClassLRModel(labelToInt: Map[String, Int],
                                   lrm: IdibonSparkMLLIBLRWrapper,
                                   featurePipeline: FeaturePipeline) extends MLModel
  with Archivable[IdibonMultiClassLRModel, IdibonMultiClassLRModelLoader] with StrictLogging {

  val intToLabel = labelToInt.map({case (label, index) => (index, label)}).toMap

  /**
    * Returns the type of model.
    *
    * @return canonical class name.
    */
  override def getType(): String = this.getClass().getName()

  /**
    * The model will use a subset of features passed in. This method
    * should return the ones used.
    *
    * @return Vector (likely SparseVector) where indices correspond to features
    *         that were used.
    */
  override def getFeaturesUsed(): Vector = {
    return lrm.getFeaturesUsed()
  }

  /**
    * The method used to predict from a vector of features.
    *
    * @param features Vector of features to use for prediction.
    * @param options  Object of predict options.
    * @return
    */
  override def predict(features: Vector, options: PredictOptions): PredictResult = {
    val results = lrm.predictProbability(features)
    logger.trace(s"input = ${features.toString} output=${results.toString}")

    val builder = new MultiLabelDocumentResultBuilder(this.getType(), labelToInt.keys.toList)
    results.foreachActive((intIndex, prob) => {
      val label: String = intToLabel(intIndex)
      builder.setProbability(label, prob.toFloat)
      builder.setMatchCount(label, 1)
    })
    // create map of string label -> probability
    if (!options.significantFeatureThreshold.isNaN()) {
      // get sig features for each label
      val indexSigFeatures = lrm.getSignificantFeatures(features, options.significantFeatureThreshold)
      indexSigFeatures.foreach({ case (intIndex, features) => {
        // get human readable versions
        val indexToHumanFeatures = featurePipeline.getHumanReadableFeature(features.map(x => x._1))
        // add the per label significant features
        builder.addSignificantFeatures(
          intToLabel(intIndex),
          features.map({case (fIntIndex, weight) => (indexToHumanFeatures(fIntIndex), weight)})
        )
      }})
    }
    builder.build()
  }

  /**
    * The method used to predict from a FULL DOCUMENT!
    *
    * The model needs to handle "featurization" here.
    *
    * @param document the JObject to pull from.
    * @param options  Object of predict options.
    * @return
    */
  override def predict(document: JObject, options: PredictOptions): PredictResult = {
    return predict(featurePipeline.apply(document), options)
  }

  /** Serializes the object within the Alloy
    *
    * Implementations are responsible for persisting any internal state
    * necessary to re-load the object (for example, feature-to-vector
    * index mappings) to the provided Alloy.Writer.
    *
    * Implementations may return a JObject of configuration data
    * to include when re-loading the object.
    *
    * @param writer destination within Alloy for any resources that
    *               must be preserved for this object to be reloadable
    * @return Some[JObject] of configuration data that must be preserved
    *         to reload the object. None if no configuration is needed
    */
  override def save(writer: Writer): Option[JObject] = {
    val coeffs = writer.within("model").resource("coefficients.libsvm")
    IdibonMultiClassLRModel.writeCodecLibSVM(
      this.labelToInt, coeffs, this.lrm.intercept, this.lrm.weights, this.lrm.numFeatures)
    coeffs.close()
    //TODO: store other model metadata like training date, etc.
    val featurePipelineMeta = featurePipeline.save(writer.within("featurePipeline"))
    Some(new JObject(List(
      JField("version", JString(IdibonMultiClassLRModel.FORMAT_VERSION)),
      JField("feature-meta", featurePipelineMeta.getOrElse(JNothing))
    )))
  }
}

/**
  * Static object that houses static functions and constants.
  */
object IdibonMultiClassLRModel extends StrictLogging {
  val FORMAT_VERSION = "0.0.1"

  /**
    * Static method to write our "libsvm" like format to a stream.
    *
    * @param labelToInt
    * @param out
    * @param intercept
    * @param coefficients
    */
  def writeCodecLibSVM(labelToInt: Map[String, Int],
                       out: DataOutputStream,
                       intercept: Double,
                       coefficients: Vector,
                       numFeatures: Int): Unit = {
    logger.info(s"Writing ${coefficients.size} dimensions with " +
      s"${coefficients.numActives} active dimensions with $intercept for MultiClass LR")
    // int to label map
    // size
    Codec.VLuint.write(out, labelToInt.size)
    labelToInt.foreach({case (label, index) => {
      // label
      Codec.String.write(out, label)
      // int index
      Codec.VLuint.write(out, index)
    }})
    // number of features
    Codec.VLuint.write(out, numFeatures)
    // intercept
    out.writeDouble(intercept)
    // dimensions
    Codec.VLuint.write(out, coefficients.size)
    // actual non-zero dimensions
    Codec.VLuint.write(out, coefficients.numActives)
    var maxCoefficient = -10000.0
    var minCoefficient = 10000.0
    coefficients.foreachActive{
      case (index, value) =>
        // do I need to worry about 0?
        Codec.VLuint.write(out, index)
        out.writeDouble(value)
        if (value > maxCoefficient) maxCoefficient = value
        if (value < minCoefficient) minCoefficient = value
    }
  }

  /**
    * Static method to read our "libsvm" like format from a stream.
    *
    * @param in
    * @return
    */
  def readCodecLibSVM(in: DataInputStream): (Double, Vector, Map[String, Int], Int) = {
    // int to label map
    val numClasses = Codec.VLuint.read(in)
    val labelToInt = (0 until numClasses).map(_ => {
      (Codec.String.read(in), Codec.VLuint.read(in))
    }).toMap
    // number of features
    val numFeatures = Codec.VLuint.read(in)
    // intercept
    val intercept = in.readDouble()
    // dimensions
    val dimensions = Codec.VLuint.read(in)
    // non-zero dimensions
    val numCoeffs = Codec.VLuint.read(in)
    val (indices, values) = (0 until numCoeffs).map { _ =>
      (Codec.VLuint.read(in), in.readDouble())
    }.unzip
    logger.info(s"Read $numCoeffs dimensions from $dimensions for Multiclass with intercept $intercept")
    (intercept, Vectors.dense(values.toArray), labelToInt, numFeatures)
  }
}

class IdibonMultiClassLRModelLoader
  extends ArchiveLoader[IdibonMultiClassLRModel] with StrictLogging {
  /** Reloads the object from the Alloy
    *
    * @param engine implementation of the Engine trait
    * @param reader location within Alloy for loading any resources
    *               previous preserved by a call to
    *               { @link com.idibon.ml.feature.Archivable#save}
    * @param config archived configuration data returned by a previous
    *               call to { @link com.idibon.ml.feature.Archivable#save}
    * @return this object
    */
  override def load(engine: Engine, reader: Reader, config: Option[JObject]): IdibonMultiClassLRModel = {
    implicit val formats = DefaultFormats
    val version = (config.get \ "version" ).extract[String]
    version match {
      case IdibonMultiClassLRModel.FORMAT_VERSION =>
        logger.info(s"Attemping to load version [v. $version] for multiclass LR.")
      case _ => throw new IOException(s"Unable to load, unhandled version [v. $version] for multiclass LR.")
    }
    val coeffs = reader.within("model").resource("coefficients.libsvm")
    val (intercept: Double,
         coefficients: Vector,
         labelToInt: Map[String, Int],
         numFeatures: Int) = IdibonMultiClassLRModel.readCodecLibSVM(coeffs)
    coeffs.close()
    val featureMeta = (config.get \ "feature-meta").extract[JObject]
    val featurePipeline = new FeaturePipelineLoader().load(
      engine, reader.within("featurePipeline"), Some(featureMeta))
    new IdibonMultiClassLRModel(
      labelToInt,
      new IdibonSparkMLLIBLRWrapper(coefficients, intercept, numFeatures, labelToInt.size),
      featurePipeline)
  }
}



