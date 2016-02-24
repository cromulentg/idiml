package com.idibon.ml.train.alloy

import java.util.Random

import com.idibon.ml.common.Engine
import com.idibon.ml.predict.Classification
import com.idibon.ml.train.datagenerator.{KClassDataFrameGeneratorBuilder, MultiClassDataFrameGeneratorBuilder, SparkDataGeneratorBuilder}
import com.idibon.ml.train.furnace.{FurnaceBuilder, MultiClassLRFurnaceBuilder, SimpleLogisticRegressionFurnaceBuilder, XValLogisticRegressionFurnaceBuilder, XValWithFPLogisticRegressionFurnaceBuilder}
import org.json4s.ShortTypeHints
import org.json4s.native.Serialization
import org.json4s.native.Serialization._

/**
  * Static object to house global defaults for Alloy Trainer builders.
  */
object BuilderDefaults {
  val classHints = List(classOf[KClass1FPBuilder], classOf[KClassKFPBuilder], classOf[MultiClass1FPBuilder], classOf[LearningCurveTrainerBuilder])
  // we need to connect all the different possible classes underneath so we can
  // create a single JSON config that gets split into the respective builders.
  implicit val formats = Serialization.formats(ShortTypeHints(classHints ++
    com.idibon.ml.train.datagenerator.BuilderDefaults.classHints ++
    com.idibon.ml.train.furnace.BuilderDefaults.classHints ++
    com.idibon.ml.train.datagenerator.scales.BuilderDefaults.classHints
  ))
}

/**
  * Trait that all alloy builders need to extend.
  */
trait AlloyTrainerBuilder {

  /**
    * Returns the alloy trainer.
 *
    * @param engine
    * @return
    */
  def build(engine: Engine): AlloyTrainer

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
  * Builder for creating the K Class 1 FP trainer.
 *
  * @param dataGenBuilder The builder that produce the data generator for input to fit models.
  * @param furnaceBuilder The builder that produces the class to fit models.
  */
case class KClass1FPBuilder(private[alloy] var dataGenBuilder: SparkDataGeneratorBuilder = new KClassDataFrameGeneratorBuilder(),
                            private[alloy] var furnaceBuilder: FurnaceBuilder[Classification] = new XValLogisticRegressionFurnaceBuilder(),
                            private[alloy] var skipGangeMetrics: Boolean = false)
  extends AlloyTrainerBuilder {
  private[alloy] var engine: Engine = null

  override def build(engine: Engine): KClass1FP = {
    this.engine = engine
    new KClass1FP(this)
  }
}


/**
  * Builder for creating the K Class K FP trainer.
  *
  * @param dataGenBuilder The builder that produce the data generator for input to fit models.
  * @param furnaceBuilder The builder that produces the class to fit models.
  */
case class KClassKFPBuilder(private[alloy] var dataGenBuilder: SparkDataGeneratorBuilder =
                              new KClassDataFrameGeneratorBuilder(),
                            private[alloy] var furnaceBuilder: FurnaceBuilder[Classification] =
                              new XValWithFPLogisticRegressionFurnaceBuilder())
  extends AlloyTrainerBuilder {
  private[alloy] var engine: Engine = null

  override def build(engine: Engine): KClassKFP = {
    this.engine = engine
    new KClassKFP(this)
  }
}

/**
  * Builder for creating the Multi-class 1 FP trainer.
 *
  * @param dataGenBuilder The builder that produce the data generator for input to fit models.
  * @param furnaceBuilder The builder that produces the class to fit models.
  */
case class MultiClass1FPBuilder(private[alloy] var dataGenBuilder: SparkDataGeneratorBuilder = new MultiClassDataFrameGeneratorBuilder(),
                                private[alloy] var furnaceBuilder: FurnaceBuilder[Classification] = new MultiClassLRFurnaceBuilder())
  extends AlloyTrainerBuilder {
  private[alloy] var engine: Engine = null

  override def build(engine: Engine): MultiClass1FP = {
    this.engine = engine
    new MultiClass1FP(this)
  }
}


/**
  * Builder for creating learning curves from k-class tasks that are mutually exclusive.
  *
  * @param trainerBuilder
  * @param numFolds
  * @param portions
  * @param foldSeed
  */
case class LearningCurveTrainerBuilder(private[alloy] var trainerBuilder: AlloyTrainerBuilder = new KClass1FPBuilder(),
                                       private[alloy] var numFolds: Int = 5,
                                       private[alloy] var portions: Array[Double] = Array[Double](0.25, 0.5, 0.625, 0.75, 0.8125, 0.875, 0.9375, 1.0),
                                       private[alloy] var foldSeed: Long = new Random().nextLong())
  extends AlloyTrainerBuilder {
  private[alloy] var engine: Engine = null

  override def build(engine: Engine): LearningCurveTrainer = {
    this.engine = engine
    new LearningCurveTrainer(this)
  }
}
