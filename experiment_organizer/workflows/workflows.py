from experiment_organizer.workflows.base import PipelineWorkflow, TreeWorkflow
from experiment_organizer.models import ObjectTypeEnum
from experiment_organizer.workflows.tasks import *


class OrganoidLinkingWorkflow(TreeWorkflow):
    def __init__(self):
        super(OrganoidLinkingWorkflow, self).__init__(
            [
                FindOverlappingRoundSegmentationsTask(),
                FindBestOverlappingSegmentationLinksTask(
                    location_field='round_id'
                ),
                LinkOverlappingSegmentationsTask(
                    location_field='round_id',
                    object_type=ObjectTypeEnum.organoid,
                    min_similariry_score=0.6
                )
            ]
        )


class CellLinkingWorkflow(TreeWorkflow):
    def __init__(self):
        super(OrganoidLinkingWorkflow, self).__init__(
            [
                FindOverlappingStackSegmentationsTask(),
                FindBestOverlappingSegmentationLinksTask(
                    location_field='z_index'
                ),
                LinkOverlappingSegmentationsTask(
                    location_field='z_index',
                    object_type=ObjectTypeEnum.cell,
                    min_similariry_score=0.75
                )
            ]
        )
