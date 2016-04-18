package com.idibon.ml.train.datagenerator.scales

import com.idibon.ml.common.{Engine, EmbeddedEngine}
import org.apache.spark.sql.{SQLContext, DataFrame}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.scalatest.{FunSpec, Matchers}

import scala.io.Source

trait TestDataBuilder {
  def sql: SQLContext
  def engine: Engine

  def generateData(classes: (Double, Int)*): DataFrame = {
    val i = classes.map({ case (label, count) => {
      Stream.continually(LabeledPoint(label, Vectors.dense(1, 1, 1, 1))).take(count)
    }}).reduce(_ ++ _).toSeq
    /* Uses numSlices = 1 to always partition the data into one set.
     * This is needed because by default, when run locally, it defaults
     * to partitioning to the number of cores. The sample method, as I
     * understand, samples on a per partition basis. Since testing
     * and the like happens on various machines, it would result in a
     * different amount of partitions (since the number of cores isn't
     * fixed) and so, with fluctuating partitions, you will get different
     * results and hence failing tests; therefore we hard code it.
     */
    sql.createDataFrame(engine.sparkContext.parallelize(i.toSeq, 1))
  }
}

/**
  * Verifies the functionality of BalancedBinaryScale class
  */
class BalancedBinaryScaleSpec extends FunSpec with Matchers
    with TestDataBuilder {

  override val engine = new EmbeddedEngine
  override val sql = new SQLContext(engine.sparkContext)

  it("should balance a dataset with too many negatives") {
    val balancer = new BalancedBinaryScaleBuilder(seed = 1L).build()
    val balanced = balancer(sql, generateData(0.0 -> 10, 1.0 -> 2))
    val dist = balanced.groupBy("label").count.orderBy("label").collect
    val results = dist.map(r => r.getAs[Double]("label") -> r.getAs[Long]("count"))
    results shouldBe List(0.0 -> 4, 1.0 -> 2)
  }

  it("should balance a dataset with too many positives") {
    val balancer = new BalancedBinaryScaleBuilder(seed = 1L).build()
    val balanced = balancer(sql, generateData(0.0 -> 2, 1.0 -> 10))
    val dist = balanced.groupBy("label").count.orderBy("label").collect
    val results = dist.map(r => r.getAs[Double]("label") -> r.getAs[Long]("count"))
    results shouldBe List(0.0 -> 2, 1.0 -> 4)
  }

  it("should not do anything to a dataset with ratio inside threshold") {
    val balancer = new BalancedBinaryScaleBuilder(seed = 1L).build()
    val rawData = generateData(0.0 -> 7, 1.0 -> 10)
    val balanced = balancer(sql, rawData)
    balanced.count shouldBe 17
    balanced should be theSameInstanceAs rawData
  }

  it("should raise an error if too many classes are present") {
    val balancer = new BalancedBinaryScaleBuilder(seed = 1L).build()
    val rawData = generateData(0.0 -> 1, 1.0 -> 1, 2.0 -> 1)
    intercept[IllegalArgumentException] { balancer(sql, rawData) }
  }
}

class DataSetScaleSpec extends FunSpec with Matchers with TestDataBuilder {

  override val engine = new EmbeddedEngine
  override val sql = new SQLContext(engine.sparkContext)

  describe("ensureTwoClasses") {

    it("should return None when we have more than 1 class.") {
      DataSetScale.ensureTwoClasses(sql, generateData(0.0 -> 5, 1.0 -> 5)) shouldBe None
    }

    describe("when only label == 1.0 exists") {
      it("should return Some with an equal number of label == 0.0 items") {
        val defended = DataSetScale.ensureTwoClasses(sql, generateData(1.0 -> 2))
        defended should not be None

        val dist = defended.get.groupBy("label").count.orderBy("label").collect
          .map(r => r.getAs[Double]("label") -> r.getAs[Long]("count"))

        dist shouldBe List(0.0 -> 2, 1.0 -> 2)
      }
    }

    describe("when only label == 0.0 exists") {
      it("should return Some with an equal number of label == 1.0 items") {
        val defended = DataSetScale.ensureTwoClasses(sql, generateData(0.0 -> 2))
        defended should not be None

        val dist = defended.get.groupBy("label").count.orderBy("label").collect
          .map(r => r.getAs[Double]("label") -> r.getAs[Long]("count"))

        dist shouldBe List(0.0 -> 2, 1.0 -> 2)
      }
    }

    describe("when only label == 2.0 exists") {
      it("should return Some with an equal number of label == 0.0 items") {
        val defended = DataSetScale.ensureTwoClasses(sql, generateData(2.0 -> 2))
        defended should not be None

        val dist = defended.get.groupBy("label").count.orderBy("label").collect
          .map(r => r.getAs[Double]("label") -> r.getAs[Long]("count"))

        dist shouldBe List(0.0 -> 2, 2.0 -> 2)
      }
    }
  }
}

class NoOpScaleSpec extends FunSpec with Matchers with TestDataBuilder {
  override val engine = new EmbeddedEngine
  override val sql = new SQLContext(engine.sparkContext)

  it("should return the same instance if multiple labels exist") {
    val balancer = new NoOpScaleBuilder().build()
    val rawData = generateData(0.0 -> 2, 1.0 -> 10)
    val balanced = balancer(sql, rawData)
    balanced should be theSameInstanceAs rawData
  }
}
