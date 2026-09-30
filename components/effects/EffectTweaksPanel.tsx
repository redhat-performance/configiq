"use client";

import React, { useState } from "react";
import {
  Drawer,
  DrawerContent,
  DrawerContentBody,
  DrawerPanelContent,
  DrawerHead,
  DrawerActions,
  DrawerCloseButton,
  Button,
  Form,
  FormGroup,
  Slider,
  Switch,
  Title,
  Divider,
  Stack,
  StackItem,
  Content,
  ExpandableSection,
} from "@patternfly/react-core";
import { CogIcon } from "@patternfly/react-icons";
import { EffectConfig, defaultEffectConfig } from "./HoverCard";

interface EffectTweaksPanelProps {
  config: EffectConfig;
  onChange: (config: EffectConfig) => void;
}

export function EffectTweaksPanel({ config, onChange }: EffectTweaksPanelProps) {
  const [isOpen, setIsOpen] = useState(false);

  const updateConfig = (key: keyof EffectConfig, value: any) => {
    onChange({ ...config, [key]: value });
  };

  const resetToDefaults = () => {
    onChange(defaultEffectConfig);
  };

  const panelContent = (
    <DrawerPanelContent style={{ width: "400px" }}>
      <DrawerHead>
        <Title headingLevel="h2" size="xl">
          <CogIcon /> Effect tweaks
        </Title>
        <DrawerActions>
          <DrawerCloseButton onClick={() => setIsOpen(false)} />
        </DrawerActions>
      </DrawerHead>
      <DrawerContentBody style={{ padding: "1.5rem", overflowY: "auto" }}>
        <Form>
          <Stack hasGutter>
            {/* Perspective Tilt */}
            <StackItem>
              <ExpandableSection toggleText="Perspective tilt" isExpanded>
                <Stack hasGutter>
                  <StackItem>
                    <FormGroup>
                      <Switch
                        id="tilt-enabled"
                        label="Enable tilt effect"
                        isChecked={config.tiltEnabled}
                        onChange={(_, checked) => updateConfig("tiltEnabled", checked)}
                      />
                    </FormGroup>
                  </StackItem>

                  <StackItem>
                    <FormGroup label={`Tilt intensity: ${config.tiltIntensity}°`}>
                      <Slider
                        value={config.tiltIntensity}
                        min={0}
                        max={50}
                        step={1}
                        onChange={(_event, value) => updateConfig("tiltIntensity", value)}
                        isDisabled={!config.tiltEnabled}
                      />
                    </FormGroup>
                  </StackItem>

                  <StackItem>
                    <FormGroup label={`Smoothness: ${config.tiltSmooth.toFixed(2)}`}>
                      <Slider
                        value={config.tiltSmooth * 100}
                        min={1}
                        max={100}
                        step={1}
                        onChange={(_event, value) => updateConfig("tiltSmooth", value / 100)}
                        isDisabled={!config.tiltEnabled}
                      />
                      <Content component="small" style={{ color: "#6a6e73" }}>
                        Lower = smoother but slower
                      </Content>
                    </FormGroup>
                  </StackItem>
                </Stack>
              </ExpandableSection>
            </StackItem>

            <Divider />

            {/* Glow Effect */}
            <StackItem>
              <ExpandableSection toggleText="Outer glow" isExpanded>
                <Stack hasGutter>
                  <StackItem>
                    <FormGroup>
                      <Switch
                        id="glow-enabled"
                        label="Enable glow"
                        isChecked={config.glowEnabled}
                        onChange={(_, checked) => updateConfig("glowEnabled", checked)}
                      />
                    </FormGroup>
                  </StackItem>

                  <StackItem>
                    <FormGroup label="Glow color">
                      <input
                        type="color"
                        value={config.glowColor}
                        onChange={(e) => updateConfig("glowColor", e.target.value)}
                        disabled={!config.glowEnabled}
                        style={{
                          width: "100%",
                          height: "36px",
                          border: "1px solid #d2d2d2",
                          borderRadius: "3px",
                          cursor: config.glowEnabled ? "pointer" : "not-allowed",
                        }}
                      />
                    </FormGroup>
                  </StackItem>

                  <StackItem>
                    <FormGroup label={`Blur intensity: ${config.glowIntensity}px`}>
                      <Slider
                        value={config.glowIntensity}
                        min={0}
                        max={100}
                        step={1}
                        onChange={(_event, value) => updateConfig("glowIntensity", value)}
                        isDisabled={!config.glowEnabled}
                      />
                    </FormGroup>
                  </StackItem>

                  <StackItem>
                    <FormGroup label={`Spread: ${config.glowSpread}px`}>
                      <Slider
                        value={config.glowSpread}
                        min={0}
                        max={50}
                        step={1}
                        onChange={(_event, value) => updateConfig("glowSpread", value)}
                        isDisabled={!config.glowEnabled}
                      />
                    </FormGroup>
                  </StackItem>
                </Stack>
              </ExpandableSection>
            </StackItem>

            <Divider />

            {/* Iridescence */}
            <StackItem>
              <ExpandableSection toggleText="Iridescence" isExpanded>
                <Stack hasGutter>
                  <StackItem>
                    <FormGroup>
                      <Switch
                        id="iridescence-enabled"
                        label="Enable iridescent overlay"
                        isChecked={config.iridescenceEnabled}
                        onChange={(_, checked) => updateConfig("iridescenceEnabled", checked)}
                      />
                    </FormGroup>
                  </StackItem>

                  <StackItem>
                    <FormGroup label={`Intensity: ${(config.iridescenceIntensity * 100).toFixed(0)}%`}>
                      <Slider
                        value={config.iridescenceIntensity * 100}
                        min={0}
                        max={100}
                        step={1}
                        onChange={(_event, value) => updateConfig("iridescenceIntensity", value / 100)}
                        isDisabled={!config.iridescenceEnabled}
                      />
                    </FormGroup>
                  </StackItem>

                  <StackItem>
                    <FormGroup label={`Hue shift: ${config.iridescenceHueShift}°`}>
                      <Slider
                        value={config.iridescenceHueShift}
                        min={0}
                        max={360}
                        step={10}
                        onChange={(_event, value) => updateConfig("iridescenceHueShift", value)}
                        isDisabled={!config.iridescenceEnabled}
                      />
                      <Content component="small" style={{ color: "#6a6e73" }}>
                        Controls color range of rainbow effect
                      </Content>
                    </FormGroup>
                  </StackItem>
                </Stack>
              </ExpandableSection>
            </StackItem>

            <Divider />

            {/* Specular Highlight */}
            <StackItem>
              <ExpandableSection toggleText="Specular highlight" isExpanded>
                <Stack hasGutter>
                  <StackItem>
                    <FormGroup>
                      <Switch
                        id="specular-enabled"
                        label="Enable specular glow"
                        isChecked={config.specularEnabled}
                        onChange={(_, checked) => updateConfig("specularEnabled", checked)}
                      />
                    </FormGroup>
                  </StackItem>

                  <StackItem>
                    <FormGroup label="Highlight color">
                      <input
                        type="color"
                        value={config.specularColor}
                        onChange={(e) => updateConfig("specularColor", e.target.value)}
                        disabled={!config.specularEnabled}
                        style={{
                          width: "100%",
                          height: "36px",
                          border: "1px solid #d2d2d2",
                          borderRadius: "3px",
                          cursor: config.specularEnabled ? "pointer" : "not-allowed",
                        }}
                      />
                    </FormGroup>
                  </StackItem>

                  <StackItem>
                    <FormGroup label={`Size: ${config.specularSize}px`}>
                      <Slider
                        value={config.specularSize}
                        min={50}
                        max={500}
                        step={10}
                        onChange={(_event, value) => updateConfig("specularSize", value)}
                        isDisabled={!config.specularEnabled}
                      />
                    </FormGroup>
                  </StackItem>

                  <StackItem>
                    <FormGroup label={`Intensity: ${(config.specularIntensity * 100).toFixed(0)}%`}>
                      <Slider
                        value={config.specularIntensity * 100}
                        min={0}
                        max={100}
                        step={1}
                        onChange={(_event, value) => updateConfig("specularIntensity", value / 100)}
                        isDisabled={!config.specularEnabled}
                      />
                    </FormGroup>
                  </StackItem>
                </Stack>
              </ExpandableSection>
            </StackItem>

            <Divider />

            {/* Noise Texture */}
            <StackItem>
              <ExpandableSection toggleText="Noise texture" isExpanded>
                <Stack hasGutter>
                  <StackItem>
                    <FormGroup>
                      <Switch
                        id="noise-enabled"
                        label="Enable noise grain"
                        isChecked={config.noiseEnabled}
                        onChange={(_, checked) => updateConfig("noiseEnabled", checked)}
                      />
                    </FormGroup>
                  </StackItem>

                  <StackItem>
                    <FormGroup label={`Opacity: ${(config.noiseOpacity * 100).toFixed(0)}%`}>
                      <Slider
                        value={config.noiseOpacity * 100}
                        min={0}
                        max={30}
                        step={1}
                        onChange={(_event, value) => updateConfig("noiseOpacity", value / 100)}
                        isDisabled={!config.noiseEnabled}
                      />
                    </FormGroup>
                  </StackItem>

                  <StackItem>
                    <FormGroup label={`Grain scale: ${config.noiseScale.toFixed(1)}`}>
                      <Slider
                        value={config.noiseScale * 10}
                        min={10}
                        max={100}
                        step={5}
                        onChange={(_event, value) => updateConfig("noiseScale", value / 10)}
                        isDisabled={!config.noiseEnabled}
                      />
                      <Content component="small" style={{ color: "#6a6e73" }}>
                        Higher = finer grain
                      </Content>
                    </FormGroup>
                  </StackItem>
                </Stack>
              </ExpandableSection>
            </StackItem>

            <Divider />

            {/* Border Glow */}
            <StackItem>
              <ExpandableSection toggleText="Border glow" isExpanded>
                <Stack hasGutter>
                  <StackItem>
                    <FormGroup>
                      <Switch
                        id="border-glow-enabled"
                        label="Enable border glow"
                        isChecked={config.borderGlowEnabled}
                        onChange={(_, checked) => updateConfig("borderGlowEnabled", checked)}
                      />
                    </FormGroup>
                  </StackItem>

                  <StackItem>
                    <FormGroup label="Border color">
                      <input
                        type="color"
                        value={config.borderGlowColor}
                        onChange={(e) => updateConfig("borderGlowColor", e.target.value)}
                        disabled={!config.borderGlowEnabled}
                        style={{
                          width: "100%",
                          height: "36px",
                          border: "1px solid #d2d2d2",
                          borderRadius: "3px",
                          cursor: config.borderGlowEnabled ? "pointer" : "not-allowed",
                        }}
                      />
                    </FormGroup>
                  </StackItem>

                  <StackItem>
                    <FormGroup label={`Width: ${config.borderGlowWidth}px`}>
                      <Slider
                        value={config.borderGlowWidth}
                        min={1}
                        max={10}
                        step={1}
                        onChange={(_event, value) => updateConfig("borderGlowWidth", value)}
                        isDisabled={!config.borderGlowEnabled}
                      />
                    </FormGroup>
                  </StackItem>
                </Stack>
              </ExpandableSection>
            </StackItem>

            <Divider />

            {/* Reset Button */}
            <StackItem>
              <Button variant="secondary" isBlock onClick={resetToDefaults}>
                Reset to defaults
              </Button>
            </StackItem>
          </Stack>
        </Form>
      </DrawerContentBody>
    </DrawerPanelContent>
  );

  return (
    <>
      {/* Floating tweaks button */}
      <div
        style={{
          position: "fixed",
          bottom: "2rem",
          right: "2rem",
          zIndex: 1000,
        }}
      >
        <Button
          variant="primary"
          icon={<CogIcon />}
          onClick={() => setIsOpen(true)}
          style={{
            boxShadow: "0 4px 12px rgba(0,0,0,0.15)",
            fontSize: "0.9375rem",
            padding: "0.75rem 1.25rem",
          }}
        >
          Effect tweaks
        </Button>
      </div>

      {/* Drawer panel */}
      <Drawer isExpanded={isOpen} position="right">
        <DrawerContent panelContent={panelContent}>
          <DrawerContentBody />
        </DrawerContent>
      </Drawer>
    </>
  );
}
