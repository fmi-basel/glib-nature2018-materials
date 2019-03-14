import os
import platform

from experiment_organizer.models import init_sqlite_engine, Base, Session
from experiment_organizer.readers import ExperimentReader
from experiment_organizer.workflows import \
    OrganoidLinkingWorkflow, CellLinkingWorkflow


class ExperimentOrganizer():
    _engine = None
    _session = None
    auto_commit = None

    _experiment_path = None
    _experiment = None

    def __init__(
        self, experiment_path, database_name=':memory:',
        auto_commit=True, mod_spatialite_path=None
    ):
        self._experiment_path = experiment_path
        self._experiment = None

        if database_name == ':memory:':
            database_path = database_name
        else:
            database_name = (
                database_name if database_name.endswith('.db')
                else database_name + ".db"
            )
            database_path = os.path.join(experiment_path, database_name)

        if platform.system() == "Windows":
            self._engine = init_sqlite_engine(
                database_path,
                mod_spatialite_path=(
                    mod_spatialite_path or
                    os.path.join(
                        os.path.dirname(__file__), 'mod_spatialite-wn'
                    )
                ),
                echo=False
            )
        elif platform.system() == "Linux":
            self._engine = init_sqlite_engine(
                database_path,
                mod_spatialite_path=(
                    mod_spatialite_path or
                    os.path.join(
                        os.path.dirname(__file__), 'mod_spatialite-lx'
                    )
                ),
                echo=False
            )
        else:
            self._engine = init_sqlite_engine(
                database_path,
                mod_spatialite_path=(
                    mod_spatialite_path or
                    os.path.join(os.path.dirname(__file__), 'mod_spatialite')
                ),
                echo=False
            )

        self.auto_commit = auto_commit

        # create all database tables
        Base.metadata.create_all(self._engine)
        self._session = Session()
        self._session.flush()
        self._session.commit()
        self._session.close()

    def __enter__(self):
        self._session = Session()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if exception_type is None and self.auto_commit:
            self._session.commit()
        else:
            self._session.rollback()
        self._session.close()
        self._engine.dispose()

    def commit(self):
        if self._session is not None:
            self._session.commit()

    def rollback(self):
        if self._session is not None:
            self._session.rollback()

    def build_index(
        self,
        load_images=True, load_segmentations=True,
        load_segmentation_features=True,
        include_image_folders=[], exclude_tags=[]
    ):
        self._experiment = ExperimentReader.read(
            self._experiment_path,
            load_images=load_images,
            load_segmentations=load_segmentations,
            load_segmentation_features=load_segmentation_features,
            include_image_folders=include_image_folders,
            exclude_tags=exclude_tags
        )
        self._experiment.add()

    def link_organoids(self):
        self.process_index(OrganoidLinkingWorkflow())

    def link_cells(self):
        self.process_index(CellLinkingWorkflow())

    def process_index(self, workflow):
        if self._experiment is None:
            raise RuntimeError(
                'No experiment found! Please build an experiment index first.'
            )
        return workflow.run(self._experiment)
