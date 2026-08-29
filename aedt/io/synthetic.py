"""SYNTHETIC dataset adapter.

Everything this adapter returns is stamped SYNTHETIC, in the frame's
``attrs``, in the ``DatasetAudit``, in every downstream result object, and on
every figure and table. There is no code path by which synthetic data can
acquire a REAL stamp.
"""
from __future__ import annotations

from pathlib import Path

from ..constants import SEED, DataStatus, DatasetRole
from ..schemas import DatasetAudit
from ..simulate.generator import cohort_to_long_frame, simulate_cohort
from .base import DatasetAdapter, LoadResult, register_adapter

__all__ = ["SyntheticAdapter"]


class SyntheticAdapter(DatasetAdapter):
    """A cohort generated from the frozen model of ROUND-17 §G."""

    name = "synthetic"
    role = DatasetRole.SIMULATION
    primary_sensor = "conversation_minutes"
    report_variable = "simulated 5-point ordinal stress item"
    can_support_longitudinal_estimand = True
    acquisition_instructions = (
        "No acquisition needed. Generated in-process from the frozen "
        "data-generating process with seed 20260828.")

    def __init__(self, *, rho: float = 0.85, n_participants: int = 48,
                 n_per_epoch: int = 200, seed: int = SEED,
                 placement: str = "skewed", phi: float = 0.4,
                 sigma_p: float = 0.8, sigma_r: float = 0.5,
                 link: str = "identity"):
        self.rho = rho
        self.n_participants = n_participants
        self.n_per_epoch = n_per_epoch
        self.seed = seed
        self.placement = placement
        self.phi = phi
        self.sigma_p = sigma_p
        self.sigma_r = sigma_r
        self.link = link

    @classmethod
    def from_config(cls, cfg) -> "SyntheticAdapter":
        return cls(
            rho=float(cfg.get("simulation.true_rho", 0.85)),
            n_participants=int(cfg.get("simulation.n_participants", 48)),
            n_per_epoch=int(cfg.get("simulation.n_per_epoch", 350)),
            seed=int(cfg.get("run.seed", SEED)),
            placement=str(cfg.get("simulation.threshold_placement", "skewed")),
            phi=float(cfg.get("simulation.sensor_ar1_phi", 0.4)),
            sigma_p=float(cfg.get("simulation.sigma_sensor", 0.8)),
            sigma_r=float(cfg.get("simulation.sigma_report", 0.5)),
            link=str(cfg.get("simulation.link", "identity")))

    def locate(self, root: Path) -> dict[str, list[Path]]:
        return {}          # generated, never read from disk

    def audit(self, root=None) -> DatasetAudit:
        res = self.load(root)
        df = res.frame
        per = df.groupby("pid").size()
        span = (df["ts"].max() - df["ts"].min()).total_seconds() / 86400.0
        return DatasetAudit(
            dataset_name=self.name, role=self.role,
            data_status=DataStatus.SYNTHETIC,
            source_status=(f"generated in-process, true rho = {self.rho}, "
                           f"seed = {self.seed}, threshold placement = "
                           f"{self.placement}"),
            local_files_available=False, root_path=None,
            participant_count=int(df["pid"].nunique()),
            observation_count=int(len(df)),
            sensor_modalities=("simulated conversation minutes",),
            self_report_variables=(self.report_variable,),
            self_report_scale="5-point ordered, severity 1 (least) .. 5 (most)",
            stress_labels=("Feeling great", "Feeling good", "A little stressed",
                           "Definitely stressed", "Stressed out"),
            raw_stored_codes=(1, 2, 3, 4, 5),
            code_to_label_mapping={1: "Feeling great", 2: "Feeling good",
                                   3: "A little stressed",
                                   4: "Definitely stressed", 5: "Stressed out"},
            code_to_severity_mapping={i: i for i in range(1, 6)},
            timestamps_present=True, timestamp_format="ISO datetime",
            timezone="naive (simulation clock)",
            longitudinal_span_days=float(span),
            observations_per_participant={str(k): int(v) for k, v in per.items()},
            median_observations_per_participant=float(per.median()),
            missingness={"report": 0.0, self.primary_sensor: 0.0},
            participant_level_coverage={str(k): 1.0 for k in per.index},
            sensor_report_alignment=("one sensor value per report occasion by "
                                     "construction"),
            conversation_context_available=True,
            eligible_for_primary_analysis=True,
            eligible_for_benchmark_analysis=False,
            acquisition_instructions=self.acquisition_instructions,
            notes=("SYNTHETIC. Not evidence about humans. The known true rho "
                   f"is {self.rho}, which is what the known-answer tests "
                   "check the estimator against.",))

    def load(self, root=None) -> LoadResult:
        cohort = simulate_cohort(
            self.rho, n_participants=self.n_participants,
            n_per_epoch=self.n_per_epoch, seed=self.seed,
            placement=self.placement, phi=self.phi, sigma_p=self.sigma_p,
            sigma_r=self.sigma_r, link=self.link)
        df = cohort_to_long_frame(cohort, sensor=self.primary_sensor)
        K = cohort[0].n_categories
        return LoadResult(frame=df, n_categories=K, sensor=self.primary_sensor,
                          data_status=DataStatus.SYNTHETIC,
                          provenance={"generator": "aedt.simulate.generator",
                                      "true_rho": self.rho, "seed": self.seed,
                                      "threshold_placement": self.placement,
                                      "sensor_ar1_phi": self.phi})


register_adapter(SyntheticAdapter())
