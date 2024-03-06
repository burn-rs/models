use core::cmp::max;

use crate::model::blocks::expand;

use super::{
    blocks::{BaseConv, BaseConvConfig, Focus, FocusConfig},
    bottleneck::{CspBottleneck, CspBottleneckConfig, SppBottleneck, SppBottleneckConfig},
};
use burn::{
    module::Module,
    tensor::{backend::Backend, Device, Tensor},
};

/// Darknet backbone feature maps.
pub struct DarknetFeatures<B: Backend>(pub Tensor<B, 4>, pub Tensor<B, 4>, pub Tensor<B, 4>);

/// [CSPDarknet-53](https://paperswithcode.com/method/cspdarknet53) backbone.
#[derive(Module, Debug)]
pub struct CspDarknet<B: Backend> {
    stem: Focus<B>,
    dark2: CspBlock<B>,
    dark3: CspBlock<B>,
    dark4: CspBlock<B>,
    dark5: CspBlock<B>,
}

impl<B: Backend> CspDarknet<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> DarknetFeatures<B> {
        let x = self.stem.forward(x);
        let x = self.dark2.forward(x);
        let f1 = self.dark3.forward(x);
        let f2 = self.dark4.forward(f1.clone());
        let f3 = self.dark5.forward(f2.clone());

        DarknetFeatures(f1, f2, f3)
    }
}

/// [CSPDarknet-53](CspDarknet) configuration.
pub struct CspDarknetConfig {
    stem: FocusConfig,
    dark2: CspBlockConfig,
    dark3: CspBlockConfig,
    dark4: CspBlockConfig,
    dark5: CspBlockConfig,
}

impl CspDarknetConfig {
    /// Create a new instance of the CSPDarknet-53 [config](CspDarknetConfig).
    pub fn new(depth: f64, width: f64) -> Self {
        assert!(
            [0.33, 0.67, 1.0, 1.33].contains(&depth),
            "invalid depth value {depth}"
        );

        assert!(
            [0.25, 0.375, 0.5, 0.75, 1.0, 1.25].contains(&width),
            "invalid width value {width}"
        );

        let base_channels = expand(64, width);
        let base_depth = max((depth * 3_f64).round() as usize, 1);

        let stem = FocusConfig::new(3, base_channels, 3, 1);
        let dark2 = CspBlockConfig::new(base_channels, base_channels * 2, base_depth, false);
        let dark3 =
            CspBlockConfig::new(base_channels * 2, base_channels * 4, base_depth * 3, false);
        let dark4 =
            CspBlockConfig::new(base_channels * 4, base_channels * 8, base_depth * 3, false);
        let dark5 = CspBlockConfig::new(base_channels * 8, base_channels * 16, base_depth, true);

        Self {
            stem,
            dark2,
            dark3,
            dark4,
            dark5,
        }
    }

    /// Initialize a new [CspDarknet](CspDarknet) module.
    pub fn init<B: Backend>(&self, device: &Device<B>) -> CspDarknet<B> {
        CspDarknet {
            stem: self.stem.init(device),
            dark2: self.dark2.init(device),
            dark3: self.dark3.init(device),
            dark4: self.dark4.init(device),
            dark5: self.dark5.init(device),
        }
    }

    /// Initialize a new [CspDarknet](CspDarknet) module with a [record](CspDarknetRecord).
    pub fn init_with<B: Backend>(&self, record: CspDarknetRecord<B>) -> CspDarknet<B> {
        CspDarknet {
            stem: self.stem.init_with(record.stem),
            dark2: self.dark2.init_with(record.dark2),
            dark3: self.dark3.init_with(record.dark3),
            dark4: self.dark4.init_with(record.dark4),
            dark5: self.dark5.init_with(record.dark5),
        }
    }
}

/// A BaseConv -> CspBottleneck block.
/// The SppBottleneck layer is only used in the last block of [CSPDarknet-53](CspDarknet).
#[derive(Module, Debug)]
pub struct CspBlock<B: Backend> {
    conv: BaseConv<B>,
    c3: CspBottleneck<B>,
    spp: Option<SppBottleneck<B>>,
}

impl<B: Backend> CspBlock<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut x = self.conv.forward(x);

        if let Some(spp) = &self.spp {
            x = spp.forward(x);
        }

        self.c3.forward(x)
    }
}

/// [CSP block](CspBlock) configuration.
pub struct CspBlockConfig {
    conv: BaseConvConfig,
    c3: CspBottleneckConfig,
    spp: Option<SppBottleneckConfig>,
}

impl CspBlockConfig {
    /// Create a new instance of the CSP block [config](CspBlockConfig).
    pub fn new(in_channels: usize, out_channels: usize, depth: usize, spp: bool) -> Self {
        let conv = BaseConvConfig::new(in_channels, out_channels, 3, 2, 1);
        let c3 = CspBottleneckConfig::new(out_channels, out_channels, depth, 0.5, !spp);

        let spp = if spp {
            Some(SppBottleneckConfig::new(out_channels, out_channels))
        } else {
            None
        };

        Self { conv, c3, spp }
    }

    /// Initialize a new [CSP block](CspBlock) module.
    pub fn init<B: Backend>(&self, device: &Device<B>) -> CspBlock<B> {
        CspBlock {
            conv: self.conv.init(device),
            c3: self.c3.init(device),
            spp: self.spp.as_ref().map(|spp| spp.init(device)),
        }
    }

    /// Initialize a new [CSP block](CspBlock) module with a [record](CspBlockRecord).
    pub fn init_with<B: Backend>(&self, record: CspBlockRecord<B>) -> CspBlock<B> {
        CspBlock {
            conv: self.conv.init_with(record.conv),
            c3: self.c3.init_with(record.c3),
            spp: self.spp.as_ref().map(|d| {
                d.init_with(
                    record
                        .spp
                        .expect("Should initialize SppBottleneck block with record."),
                )
            }),
        }
    }
}
