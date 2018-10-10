package sat.dpll

import math._
import java.io.File
import scala.io.Source
import scala.annotation._
import scala.util.Random

object Utils {
  def getListOfFiles(d: File, ext: String): List[File] = {
    if (d.exists && d.isDirectory) d.listFiles.filter(_.isFile).filter(_.getName.endsWith(ext)).toList
    else List[File]()
  }
  def getListOfFiles(dirPath: String, ext: String): List[File] = getListOfFiles(new File(dirPath), ext)

  def getCNFFromFolder(dir: String): List[String] = getListOfFiles(dir, "cnf").map(_.getPath)
  def getCNFFromFolder(d: File): List[String] = getListOfFiles(d, "cnf").map(_.getPath)
}

trait CNF {
  type Asn
  type Lit
  type Clause
  type Formula
}

object CNFImp extends CNF {
  type Asn = Map[Int, Boolean]
  type Lit = Int
  case class Clause(xs: List[Lit]) {
    def size = xs.size
    def contains(v: Lit): Boolean = xs.contains(v)
    def remove(v: Lit): Clause = Clause(xs.filter(_ != v))
    def containsAnyOf(vs: List[Lit]): Boolean = {
      for (v <- vs) { if (xs.contains(v)) return true }
      return false
    }
    def removeAllOccur(vs: List[Lit]): Clause = Clause(xs.filter(!vs.contains(_)))

    def assign(vb: (Lit, Boolean)): Option[Clause] = assign(vb._1, vb._2)
    def assign(v: Lit, b: Boolean): Option[Clause] = {
      var new_xs = List[Lit]()
      for (x <- xs) {
        if (abs(x) == abs(v)) { if ((x > 0) == b) return None } // This clause is satisfied.
        else { new_xs = x::new_xs }
      }
      Some(Clause(new_xs))
    }

    override def toString = s"(${xs.mkString(" ∨ ")})"
  }

  case class Formula(cs: List[Clause]) {
    lazy val nClauses = cs.size
    lazy val nVars = allVars.groupBy(abs).size

    lazy val allVars = cs.flatMap(_.xs).toSet
    lazy val unitVars = cs.filter(_.size == 1).map(_.xs.head).toSet.toList
    lazy val pureVars = allVars.groupBy(abs).filter(_._2.size==1).values.flatten.toList

    private def varsToAssignment(vars: Iterable[Lit]) =
      vars.map((x:Lit) => if (x>0) (x→true) else (-x→false)).toMap

    def addClause(c: Clause): Formula = Formula(c::cs)
    def addSingletonClause(x: Int): Formula = Formula(Clause(List(x))::cs)

    def isEmpty: Boolean = cs.isEmpty
    def hasUnsatClause: Boolean = cs.contains(Clause(List()))

    def hasUnitClause: Boolean = unitVars.size != 0
    def elimSingleUnit: (Formula, Asn) = {
      val v = unitVars(0)
      val asnmt = if (v > 0) Map(v → true) else Map(-v → false)
      val result = for (c <- cs if !c.contains(v)) yield c.remove(-v)
      (Formula(result), asnmt)
    }

    def elimUnit: (Formula, Asn) = {
      if (unitVars.groupBy(abs).exists(_._2.size == 2))
        return (Formula(List(Clause(List()))), Map())
      val asnmt = varsToAssignment(unitVars)
      val result = for { c <- cs if !c.containsAnyOf(unitVars) }
                   yield c.removeAllOccur(unitVars.map(-_))
      (Formula(result), asnmt)
    }

    def hasPureClause: Boolean = pureVars.size != 0
    def elimPure: (Formula, Asn) = {
      val asnmt = varsToAssignment(pureVars)
      val result = cs.filter(!_.containsAnyOf(pureVars))
      (Formula(result), asnmt)
    }

    def pickFirst: Lit = cs.head.xs.head
    def pickRandom: Lit = Random.shuffle(cs.head.xs).head

    def assign(vb: (Int, Boolean)): Formula = assign(vb._1, vb._2)
    def assign(v: Lit, b: Boolean): Formula =
      Formula((for (c <- cs) yield c.assign(v, b)).filter(_.nonEmpty).map(_.get))

    override def toString = s"(${cs.mkString(" ∧ ")}"
  }
}

import CNFImp._

object DIMACSParser {
  def parseLines(lines: Iterator[String]): Formula = {
    val cs = for (line <- lines if
                  !(line.startsWith("c") || line.startsWith("p") ||
                      line.startsWith("0") || line.startsWith("%") || line.isEmpty)) yield {
      val ns = line.split(" ").filter(_.nonEmpty).map(_.toInt).toList
      assert(ns.last == 0)
      Clause(ns.dropRight(1))
    }
    Formula(cs.toList)
  }

  def parse(input: String): Formula = parseLines(input.split("\\r?\\n").iterator)

  def parseFromResource(filePath: String): Formula = parseLines(Source.fromResource(filePath).getLines)

  def parseFromPath(filePath: String): Formula = parseLines(Source.fromFile(filePath).getLines)
}

trait Solver {
  def solve(f: Formula): Option[Asn]
}

case object DPLLNaive extends Solver {
  /* A naive DPLL implementation just uses unit propogation
   * for single variables once a time; the pure variables
   * elimination and assignment are implemented by adding
   * a new unit clause.
   */
  def dpll_naive(f: Formula, assgn: Asn): Option[Asn] = {
    if (f.hasUnsatClause) return None
    if (f.isEmpty) return Some(assgn)
    if (f.hasUnitClause) {
      val (new_f, new_assgn) = f.elimSingleUnit
      return dpll_naive(new_f, assgn ++ new_assgn)
    }
    if (f.hasPureClause) return dpll_naive(f.addSingletonClause(f.pureVars(0)), assgn)
    val v = f.pickFirst
    val tryTrue = dpll_naive(f.addSingletonClause(v), assgn)
    if (tryTrue.nonEmpty) tryTrue
    else dpll_naive(f.addSingletonClause(-v), assgn)
  }

  def solve(f: Formula): Option[Asn] = dpll_naive(f, Map[Int, Boolean]()) match {
    case Some(m) => Some(m.map({ case (v, b) => if (v < 0) (-v, !b) else (v, b) }))
    case None => None
  }
}

case object DPLL extends Solver {
  /* This DPLL uses multi-variable unit propogation, as well as
   * pure variable elimination on multi-variables.
   * The assignment is implemented as eliminating a variable
   * in a clause.
   */
  def dpll(f: Formula, assgn: Asn): Option[Asn] = {
    if (f.hasUnsatClause) return None
    if (f.isEmpty) return Some(assgn)
    if (f.hasUnitClause) {
      val (new_f, new_assgn) = f.elimUnit
      return dpll(new_f, assgn ++ new_assgn)
    }
    if (f.hasPureClause) {
      val (new_f, new_assgn) = f.elimPure
      return dpll(new_f, assgn ++ new_assgn)
    }
    val v = f.pickFirst
    val tryTrue = dpll(f.assign(v→true), assgn+(v→true))
    if (tryTrue.nonEmpty) tryTrue
    else dpll(f.assign(v→false), assgn+(v→false))
  }

  def solve(f: Formula): Option[Asn] = dpll(f, Map[Int, Boolean]()) match {
    case Some(m) => Some(m.map({ case (v, b) => if (v < 0) (-v, !b) else (v, b) }))
    case None => None
  }
}

case object DPLLCPS extends Solver {
  /* The CPS implementation of `dpll_naive`. */
  type Cont = () ⇒ Option[Asn]
  def dpll_naive_cps(f: Formula, assgn: Asn, fc: Cont): Option[Asn] = {
    if (f.hasUnsatClause) return fc()
    if (f.isEmpty) return Some(assgn)
    if (f.hasUnitClause) {
      val (new_f, new_assgn) = f.elimSingleUnit
      dpll_naive_cps(new_f, assgn ++ new_assgn, fc)
    }
    else if (f.hasPureClause) {
      dpll_naive_cps(f.addSingletonClause(f.pureVars(0)), assgn, fc)
    }
    else {
      val v = f.pickFirst
      dpll_naive_cps(f.assign(v→true), assgn+(v→true), () ⇒ dpll_naive_cps(f.assign(v→false), assgn+(v→false), fc))
    }
  }

  def solve(f: Formula): Option[Asn] = dpll_naive_cps(f, Map[Int, Boolean](), () => None) match {
    case Some(m) => Some(m.map({ case (v, b) => if (v < 0) (-v, !b) else (v, b) }))
    case None => None
  }
}

case object DefuncDPLL extends Solver {
  /* Defunctionalized DPLL */
  type DeCont = List[(Lit, Formula, Asn)]
  case class State(f: Formula, assgn: Asn, fc: DeCont)

  def applyBacktrack(s: State): State = s.fc match {
    case (v, f, assgn)::tl ⇒
      State(f.assign(v→false), assgn+(v→false), tl)
  }
  def applyUnit(s: State): State = s match { case State(f, assgn, fc) =>
    val (new_f, new_assgn) = f.elimSingleUnit
    State(new_f, new_assgn, fc)
  }
  def applyPure(s: State): State = s match { case State(f, assgn, fc) =>
    State(f.addSingletonClause(f.pureVars(0)), assgn, fc)
  }

  def ddpll_navie_step(s: State): State = s match { case State(f, assgn, fc) =>
    if (f.hasUnsatClause) applyBacktrack(s)
    else if (f.hasUnitClause) applyUnit(s)
    else if (f.hasPureClause) applyPure(s)
    else {
      val v = f.pickFirst
      State(f.assign(v→true), assgn+(v→true), (v, f, assgn)::fc)
    }
  }
  def drive(s: State): Option[Asn] = s match {
    case State(f, asn, fc)  if f.isEmpty => Some(asn)
    case State(f, asn, Nil) if f.hasUnsatClause => None
    case s => drive(ddpll_navie_step(s))
  }
  def inject(f: Formula): State = State(f, Map[Int, Boolean](), List())

  def solve(f: Formula): Option[Asn] = drive(inject(f)) match {
    case Some(m) => Some(m.map({ case (v, b) => if (v < 0) (-v, !b) else (v, b) }))
    case None => None
  }
}

case object NdDefuncDPLL extends Solver {
  case class State(f: Formula, assgn: Asn)

  def applyUnit(s: State): State = s match { case State(f, assgn) =>
    val (new_f, new_assgn) = f.elimSingleUnit
    State(new_f, new_assgn)
  }
  def applyPure(s: State): State = s match { case State(f, assgn) =>
    State(f.addSingletonClause(f.pureVars(0)), assgn)
  }

  def ddpll_navie_step(s: State): Set[State] = s match { case State(f, assgn) =>
    if (f.hasUnsatClause) Set()
    else if (f.hasUnitClause) Set(applyUnit(s))
    else if (f.hasPureClause) Set(applyPure(s))
    else {
      val v = f.pickFirst
      Set(State(f.assign(v→true), assgn+(v→true)), State(f.assign(v→false), assgn+(v→false)))
    }
  }

  def drive(todo: Set[State]): Option[Asn] =
    if (todo.isEmpty) None
    else todo.head match {
      case State(f, asn) if f.isEmpty ⇒ Some(asn)
      case s ⇒ drive(todo.tail ++ ddpll_navie_step(s))
    }

  def inject(f: Formula): State = State(f, Map[Int, Boolean]())

  def solve(f: Formula): Option[Asn] = drive(Set(inject(f)))

}

object CNFExamples extends {
  /** (x1 ∨ x2 ∨ -x3 ∨ x6) ∧
    * (-x2 ∨ x4) ∧
    * (-x1 ∨ -x5) ∧
    * (-x1 ∨ x2 ∨ x3 ∨ -x4 ∨ x5) ∧
    * (-x3) ∧ (x2 ∨ x5) ∧
    * (-x5) ∧ (x1)
    */
  val example_1 = Formula(List(Clause(List(1, 2, -3, 6)),
                               Clause(List(-2, 4)),
                               Clause(List(-1, -5)),
                               Clause(List(-1, 2, 3, -4, 5)),
                               Clause(List(-3)), Clause(List(2, 5)),
                               Clause(List(-5)), Clause(List(1))))

  val example_2 = Formula(List(Clause(List(1)),
                               Clause(List(-1)),
                               Clause(List(1, 2))))
}

object DPLLTest extends App {
  import Utils._
  import DIMACSParser._
  import CNFExamples._

  //println("DPLL")
  //println(example_f.elimUnit._1.elimUnit)
  //println(example_f.elimPure)
  //println(example_f.assign(6 → true))
  //println(example_f.assign(6 → false))
  //println(solve(example_1))
  //println(solve(example_2))
  //println(Clause(List(1,2,3,4,-1,-2,-3)).removeAllOccur(List(1,2,4)))

  //val cnf3 = parseFromResource("uf20-91/uf20-010.cnf") //SAT
  //println(solve(cnf3))

  /*
  val uuf100: List[String] = getCNFFromFolder("src/main/resources/uuf100-430")
  for (f <- uuf100) {
    println(f)
    val cnf = parseFromPath(f)
    assert(solve(cnf).isEmpty)
  }
  val uf100: List[String] = getCNFFromFolder("src/main/resources/uf100-430")
  for (f <- uf100) {
    println(f)
    val cnf = parseFromPath(f)
    assert(solve(cnf).nonEmpty)
  }
  */

  //val solve = DefuncDPLL.solve _
  val solve = NdDefuncDPLL.solve _
  val uf50: List[String] = getCNFFromFolder("src/main/resources/uf50-218")
  for (f <- uf50) {
    println(f)
    val cnf = parseFromPath(f)
    assert(solve(cnf).nonEmpty)
  }

  val uuf50: List[String] = getCNFFromFolder("src/main/resources/uuf50-218")
  for (f <- uuf50) {
    println(f)
    val cnf = parseFromPath(f)
    assert(solve(cnf).isEmpty)
  }

  /*
  val uuf200: List[String] = getCNFFromFolder("src/main/resources/uuf200-860").take(50)
  for (f <- uuf200) {
    println(f)
    val cnf = parseFromPath(f)
    assert(solve(cnf).isEmpty)
  }
  val uf200: List[String] = getCNFFromFolder("src/main/resources/uf200-860").take(50)
  for (f <- uf200) {
    println(f)
    val cnf = parseFromPath(f)
    assert(solve(cnf).nonEmpty)
  }
   */
}
