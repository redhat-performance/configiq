import type { Metadata } from "next";
import { Red_Hat_Display, Red_Hat_Mono, Red_Hat_Text } from "next/font/google";
import "@patternfly/react-core/dist/styles/base.css";
// Layouts
import "@patternfly/react-styles/css/layouts/Grid/grid.css";
import "@patternfly/react-styles/css/layouts/Flex/flex.css";
import "@patternfly/react-styles/css/layouts/Stack/stack.css";
import "@patternfly/react-styles/css/layouts/Split/split.css";
import "@patternfly/react-styles/css/layouts/Gallery/gallery.css";
import "@patternfly/react-styles/css/layouts/Bullseye/bullseye.css";
// Shell components (Page, Masthead, Nav)
import "@patternfly/react-styles/css/components/Page/page.css";
import "@patternfly/react-styles/css/components/Masthead/masthead.css";
import "@patternfly/react-styles/css/components/Brand/brand.css";
import "@patternfly/react-styles/css/components/Nav/nav.css";
// Content & typography
import "@patternfly/react-styles/css/components/Title/title.css";
import "@patternfly/react-styles/css/components/Content/content.css";
// Components used across the app
import "@patternfly/react-styles/css/components/Accordion/accordion.css";
import "@patternfly/react-styles/css/components/Alert/alert.css";
import "@patternfly/react-styles/css/components/Backdrop/backdrop.css";
import "@patternfly/react-styles/css/components/Button/button.css";
import "@patternfly/react-styles/css/components/Card/card.css";
import "@patternfly/react-styles/css/components/DescriptionList/description-list.css";
import "@patternfly/react-styles/css/components/ExpandableSection/expandable-section.css";
import "@patternfly/react-styles/css/components/Form/form.css";
import "@patternfly/react-styles/css/components/FormControl/form-control.css";
import "@patternfly/react-styles/css/components/HelperText/helper-text.css";
import "@patternfly/react-styles/css/components/Label/label.css";
import "@patternfly/react-styles/css/components/ModalBox/modal-box.css";
import "@patternfly/react-styles/css/components/Popover/popover.css";
import "@patternfly/react-styles/css/components/Progress/progress.css";
import "@patternfly/react-styles/css/components/Slider/slider.css";
import "@patternfly/react-styles/css/components/Spinner/spinner.css";
import "@patternfly/react-styles/css/components/Switch/switch.css";
import "@patternfly/react-styles/css/components/Tile/tile.css";
import "@patternfly/react-styles/css/components/ToggleGroup/toggle-group.css";
import "@patternfly/react-styles/css/components/Tooltip/tooltip.css";
import "./globals.css";
import "./theme.css";
import { AppShell } from "@/components/layout/AppShell";
import { RecommendProvider } from "@/contexts/RecommendContext";
import { SettingsProvider } from "@/contexts/SettingsContext";
import { CostAssumptionsProvider } from "@/contexts/CostAssumptionsContext";

const redHatDisplay = Red_Hat_Display({
  subsets: ["latin"],
  variable: "--font-display",
});

const redHatText = Red_Hat_Text({
  subsets: ["latin"],
  variable: "--font-body",
});

const redHatMono = Red_Hat_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
});

export const metadata: Metadata = {
  title: "ConfigIQ — LLM inference sizing and cost calculator",
  description:
    "Estimate GPU requirements, compare costs, and model LLM inference economics across cloud and on-premise deployments.",
  icons: {
    icon: "/config-iq-logo.svg",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={`${redHatDisplay.variable} ${redHatText.variable} ${redHatMono.variable}`}>
        <RecommendProvider>
          <SettingsProvider>
            <CostAssumptionsProvider>
              <AppShell>{children}</AppShell>
            </CostAssumptionsProvider>
          </SettingsProvider>
        </RecommendProvider>
      </body>
    </html>
  );
}
