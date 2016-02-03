package com.idibon.ml.train.furnace

import com.idibon.ml.common.Engine
import org.json4s.ShortTypeHints
import org.json4s.native.Serialization
import org.json4s.native.Serialization.{writePretty}

/**
  * Static object with global defaults for builders.
  */
object BuilderDefaults {
  // LogisticRegression Defaults
  val TOLERANCE = 1e-4
  val REGULARIZATION_PARAMETER = 0.001
  val MAX_ITERATIONS = 100
  val ELASTIC_NET_PARAMETER = 0.9

  // parameter search related defaults
  val NUMBER_OF_FOLDS = 10
  val TOLERANCES = Array(TOLERANCE)
  val REGULARIZATION_PARAMETERS = Array(REGULARIZATION_PARAMETER, 0.01, 0.1)
  val ELASTIC_NET_PARAMETERS = Array(ELASTIC_NET_PARAMETER, 1.0)

  val classHints = List(
    classOf[MultiClassLRFurnaceBuilder],
    classOf[SimpleLogisticRegressionBuilder],
    classOf[XValLogisticRegressionBuilder])
  /* This enables us to not do Reflection Foo when trying to create a builder from JSON.
       The only requirement is that has a 'jsonClass' field with one of the names of the classes
       below.*/
  implicit val formats = Serialization.formats(ShortTypeHints(classHints))
}

/**
  * Base FuranceBuilder Trait.
  *
  * Enforces what kind of convention we want to follow.
  *
  */
trait FurnaceBuilder {

  private[furnace] var engine: Engine = null
  /**
    * Each builder needs to have a one of these that takes an engine and
    * returns a furnace
    *
    * @param engine
    * @return
    */
  def build(engine: Engine): Furnace

  /**
    * Creates a pretty printed JSON string.
    *
    * This will be useful for building a tool to output some nice JSON configuration.
    *
    * @return
    */
  override def toString(): String = {
    implicit val formats = BuilderDefaults.formats
    writePretty(this)
  }
}

/**
  * Furnaces that create a single LR model should implement this trait.
  *
  */
trait LogisticRegressionBasedFurnaceBuilder[B] extends FurnaceBuilder {

  def setMaxIterations(maxIterations: Int): B

  def setRegParam(regParam: Double): B

  def setTolerance(tolerance: Double): B

  def setElasticNetParam(elasticNetParam: Double): B
}

/**
  * Furnaces that search over a bunch of parameters and output a single LR model should
  * implement this trait.
  *
  */
trait LogisticRegressionParameterSearchBasedFurnaceBuilder[B] extends FurnaceBuilder {

  def setMaxIterations(maxIterations: Int): B

  def setRegParams(regParams: Array[Double]): B

  def setTolerances(tolerances: Array[Double]): B

  def setElasticNetParams(elasticNetParams: Array[Double]): B

  def setNumberOfFolds(numFolds: Int): B
}

/**
  * Builder class to make it easy to create the right SimpleLogisticRegression Furnace.
  *
  * TODO: Doc for each method to explain what is happening.
  *
  * @param maxIterations
  * @param regParam
  * @param tolerance
  * @param elasticNetParam
  */
case class SimpleLogisticRegressionBuilder(private[furnace] var maxIterations: Int = BuilderDefaults.MAX_ITERATIONS,
                                           private[furnace] var regParam: Double = BuilderDefaults.REGULARIZATION_PARAMETER,
                                           private[furnace] var tolerance: Double = BuilderDefaults.TOLERANCE,
                                           private[furnace] var elasticNetParam: Double = BuilderDefaults.ELASTIC_NET_PARAMETER)
  extends LogisticRegressionBasedFurnaceBuilder[SimpleLogisticRegressionBuilder]{

  override def setElasticNetParam(elasticNetParam: Double): SimpleLogisticRegressionBuilder = {
    this.elasticNetParam = elasticNetParam
    this
  }
  override def setMaxIterations(maxIterations: Int): SimpleLogisticRegressionBuilder = {
    this.maxIterations = maxIterations
    this
  }

  override def setTolerance(tolerance: Double): SimpleLogisticRegressionBuilder = {
    this.tolerance = tolerance
    this
  }

  override def setRegParam(regParam: Double): SimpleLogisticRegressionBuilder = {
    this.regParam = regParam
    this
  }
  /**
    * Each builder needs to have a one of these that takes an engine and
    * returns a furnace
    *
    * @param engine
    * @return
    */
  override def build(engine: Engine): SimpleLogisticRegression = {
    this.engine = engine
    new SimpleLogisticRegression(this)
  }
}

/**
  * Builder class for creating a MultiClassLRFurnace.
  *
  * @param maxIterations
  * @param regParam
  * @param tolerance
  */
case class MultiClassLRFurnaceBuilder(private[furnace] var maxIterations: Int = BuilderDefaults.MAX_ITERATIONS,
                                      private[furnace] var regParam: Double = BuilderDefaults.REGULARIZATION_PARAMETER,
                                      private[furnace] var tolerance: Double = BuilderDefaults.TOLERANCE)
  extends LogisticRegressionBasedFurnaceBuilder[MultiClassLRFurnaceBuilder] {

  override def setMaxIterations(maxIterations: Int): MultiClassLRFurnaceBuilder = {
    this.maxIterations = maxIterations
    this
  }

  override def setTolerance(tolerance: Double): MultiClassLRFurnaceBuilder = {
    this.tolerance = tolerance
    this
  }

  override def setElasticNetParam(elasticNetParam: Double): MultiClassLRFurnaceBuilder = {
    throw new NotImplementedError("Multi class LR Furnace does not take elastic net parameter")
  }

  override def setRegParam(regParam: Double): MultiClassLRFurnaceBuilder = {
    this.regParam = regParam
    this
  }

  /**
    * Each builder needs to have a one of these that takes an engine and
    * returns a furnace
    *
    * @param engine
    * @return
    */
  override def build(engine: Engine): MultiClassLRFurnace = {
    this.engine = engine
    new MultiClassLRFurnace(this)
  }
}

/**
  * Builder class to make it easy to create the right XValLR Furnace.
  *
  * TODO: Doc for each method to explain what is happening.
  *
  */
case class XValLogisticRegressionBuilder(private[furnace] var maxIterations: Int = BuilderDefaults.MAX_ITERATIONS,
                                         private[furnace] var regParams: Array[Double] = BuilderDefaults.REGULARIZATION_PARAMETERS,
                                         private[furnace] var tolerances: Array[Double] = BuilderDefaults.TOLERANCES,
                                         private[furnace] var elasticNetParams: Array[Double] = BuilderDefaults.ELASTIC_NET_PARAMETERS,
                                         private[furnace] var numFolds: Int = BuilderDefaults.NUMBER_OF_FOLDS)
  extends LogisticRegressionParameterSearchBasedFurnaceBuilder[XValLogisticRegressionBuilder]{

  override def setMaxIterations(maxIterations: Int): XValLogisticRegressionBuilder = {
    this.maxIterations = maxIterations
    this
  }

  override def setRegParams(regParams: Array[Double]): XValLogisticRegressionBuilder = {
    this.regParams = regParams
    this
  }

  override def setElasticNetParams(elasticNetParams: Array[Double]): XValLogisticRegressionBuilder = {
    this.elasticNetParams = elasticNetParams
    this
  }

  def setNumberOfFolds(numberOfFolds: Int): XValLogisticRegressionBuilder = {
    this.numFolds = numberOfFolds
    this
  }

  override def setTolerances(tolerances: Array[Double]): XValLogisticRegressionBuilder = {
    this.tolerances = tolerances
    this
  }

  override def build(engine: Engine): XValLogisticRegression = {
    this.engine = engine
    new XValLogisticRegression(this)
  }
}
