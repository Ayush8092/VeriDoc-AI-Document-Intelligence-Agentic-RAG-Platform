import type { Metadata } from "next";
import "./globals.css";
import AppShell from "@/components/app-shell";
import { AuthProvider } from "@/components/auth-provider";

export const metadata: Metadata = {
  title: "Veridoc — Document Intelligence",
  description:
    "Ask questions and get answers grounded in your documents, with page- and table-level citations.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className="h-full">
      <body className="min-h-full">
        <AuthProvider>
          <AppShell>{children}</AppShell>
        </AuthProvider>
      </body>
    </html>
  );
}
