"use client";

import React from "react";
import {
  Page,
  PageSidebar,
  PageSidebarBody,
  PageToggleButton,
  Masthead,
  MastheadMain,
  MastheadBrand,
  MastheadContent,
  MastheadToggle,
  Nav,
  NavList,
  NavItem,
} from "@patternfly/react-core";
import BarsIcon from "@patternfly/react-icons/dist/esm/icons/bars-icon";
import GithubIcon from "@patternfly/react-icons/dist/esm/icons/github-icon";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  HomeIcon,
  BoltIcon,
  CalculatorIcon,
  SlidersHIcon,
  CubesIcon,
  HistoryIcon,
  DollarSignIcon,
  RouteIcon,
  CogIcon,
  DatabaseIcon,
  ListIcon,
} from "@patternfly/react-icons";
import { getVersionString, getBuildTimeString, getShortCommit } from "@/lib/version";
import { useSettings } from "@/contexts/SettingsContext";

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const { costingsEnabled } = useSettings();

  const masthead = (
    <Masthead display={{ default: 'inline' }} style={{ backgroundColor: "#1a1a1a", borderBottom: "1px solid #2d2d2d" }}>
      <MastheadMain>
        <MastheadToggle>
          <PageToggleButton
            variant="plain"
            aria-label="Navigation"
            id="nav-toggle"
            style={{ width: 44, height: 44, padding: 8 }}
          >
            <BarsIcon color="white" style={{ width: 24, height: 24 }} />
          </PageToggleButton>
        </MastheadToggle>
        <MastheadBrand>
          <Link
            href="/"
            style={{
              textDecoration: "none",
              display: "flex",
              alignItems: "center",
              gap: 12,
              padding: "0 1rem",
            }}
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src="/config-iq-logo.svg" alt="" width={42} height={42} aria-hidden="true" />
            <span
              style={{
                fontFamily: "var(--font-display)",
                fontWeight: 700,
                fontSize: "1.5rem",
                color: "white",
                letterSpacing: "-0.02em",
              }}
            >
              Config<span style={{ color: "#4dabf7" }}>IQ</span>
            </span>
            <span style={{
              fontSize: '0.75rem',
              fontWeight: 500,
              color: 'rgba(255,255,255,0.5)',
              letterSpacing: '0.01em',
              marginLeft: 16,
            }}>LLM inference sizing</span>
          </Link>
        </MastheadBrand>
      </MastheadMain>
      <MastheadContent>
        <a href="https://github.com/redhat-performance/configiq" target="_blank" rel="noopener" aria-label="GitHub repository" style={{ color: 'rgba(255,255,255,0.7)', display: 'flex', padding: 8, marginLeft: 'auto' }}>
          <GithubIcon style={{ width: 21, height: 21 }} />
        </a>
      </MastheadContent>
    </Masthead>
  );

  const navItemStyle = (isActive: boolean) => ({
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    fontSize: '14.5px',
    fontWeight: 500,
    fontFamily: '"Plus Jakarta Sans", system-ui, sans-serif',
    color: isActive ? '#ffffff' : 'rgba(255,255,255,0.85)',
    backgroundColor: isActive ? '#0066cc' : 'transparent',
    padding: '9px 16px',
    textDecoration: 'none',
    borderBottom: 'none',
  });

  const iconStyle = { width: '15px', height: '15px' };

  const NavItemWithIcon = ({
    icon: Icon,
    label,
    href,
    isActive
  }: {
    icon: any;
    label: string;
    href: string;
    isActive: boolean;
  }) => (
    <NavItem
      isActive={isActive}
      to={href}
      component={(props: any) => (
        <Link {...props} style={navItemStyle(isActive)}>
          <Icon style={iconStyle} />
          <span>{label}</span>
        </Link>
      )}
    />
  );

  const groupLabelStyle = {
    padding: '12px 16px 3px',
    fontSize: '11px',
    fontWeight: 600,
    letterSpacing: '0.08em',
    textTransform: 'uppercase' as const,
    color: 'rgba(255,255,255,0.55)',
    fontFamily: '"Plus Jakarta Sans", system-ui, sans-serif',
  };

  const sidebar = (
    <PageSidebar
      style={{
        backgroundColor: "#1a1a1a",
        borderRight: "1px solid #2d2d2d",
      }}
    >
      <PageSidebarBody>
        <Nav aria-label="Main navigation" style={{
          '--pf-v6-c-nav__link--after--BorderColor': 'transparent',
          '--pf-v6-c-nav__link--after--BorderWidth': '0'
        } as React.CSSProperties}>
          <NavList style={{
            '--pf-v6-c-nav__item--after--BorderColor': 'transparent',
            '--pf-v6-c-nav__item--after--BorderWidth': '0'
          } as React.CSSProperties}>
            <NavItemWithIcon
              icon={HomeIcon}
              label="Home"
              href="/"
              isActive={pathname === "/"}
            />

            <div style={groupLabelStyle}>ESTIMATE</div>

            <NavItemWithIcon
              icon={SlidersHIcon}
              label="Recommend sizing"
              href="/recommend"
              isActive={pathname === "/recommend"}
            />
            <NavItemWithIcon
              icon={CalculatorIcon}
              label="KV cache calculator"
              href="/kv-cache"
              isActive={pathname === "/kv-cache"}
            />
            <NavItemWithIcon
              icon={BoltIcon}
              label="Predict performance"
              href="/predict"
              isActive={pathname === "/predict"}
            />
            <NavItemWithIcon
              icon={CubesIcon}
              label="GPU explorer"
              href="/gpu-explorer"
              isActive={pathname === "/gpu-explorer"}
            />
            {false && <NavItemWithIcon
              icon={ListIcon}
              label="Compare"
              href="/compare"
              isActive={pathname === "/compare"}
            />}
            {false && <NavItemWithIcon
              icon={HistoryIcon}
              label="History"
              href="#"
              isActive={false}
            />}

            {costingsEnabled && <>
            <div style={groupLabelStyle}>COSTINGS</div>

            <NavItemWithIcon
              icon={DollarSignIcon}
              label="Hybrid savings"
              href="/hybrid-savings"
              isActive={pathname === "/hybrid-savings"}
            />
            <NavItemWithIcon
              icon={CubesIcon}
              label="Cluster cost"
              href="/cluster-cost"
              isActive={pathname === "/cluster-cost"}
            />
            <NavItemWithIcon
              icon={RouteIcon}
              label="Routing economics"
              href="/routing"
              isActive={pathname === "/routing"}
            />
            </>}

            <div style={groupLabelStyle}>SYSTEM</div>

            <NavItemWithIcon
              icon={CogIcon}
              label="Settings"
              href="/settings"
              isActive={pathname === "/settings"}
            />
            {costingsEnabled && (
              <NavItemWithIcon
                icon={DatabaseIcon}
                label="Sources"
                href="/sources"
                isActive={pathname === "/sources"}
              />
            )}
          </NavList>
        </Nav>

        {/* Version Footer */}
        <div style={{
          position: 'absolute',
          bottom: 0,
          left: 0,
          right: 0,
          padding: '12px 16px',
          borderTop: '1px solid #2d2d2d',
          fontSize: '11px',
          fontFamily: 'var(--font-mono)',
          color: 'rgba(255,255,255,0.4)',
          lineHeight: '1.4'
        }}>
          <div>{getVersionString()} · {getShortCommit()}</div>
          <div style={{ fontSize: '10px', marginTop: '2px' }}>
            {getBuildTimeString()}
          </div>
        </div>
      </PageSidebarBody>
    </PageSidebar>
  );

  return (
    <Page
      masthead={masthead}
      sidebar={sidebar}
      isManagedSidebar
      defaultManagedSidebarIsOpen={true}
      style={{ backgroundColor: "#f5f5f5" }}
    >
      {children}
    </Page>
  );
}
