from class_resolver import Resolver
from dataset.space.sampling_space import BasedSamplingSpace, FastSamplingSpace

sampling_resolver = Resolver(
    {
        BasedSamplingSpace,
        FastSamplingSpace,
    },
    base=BasedSamplingSpace,  # type: ignore
    default=BasedSamplingSpace,
    suffix='SamplingSpace',
)


