/**
 * Root layout: global styles and default metadata for the diabetes demo UI.
 */
import "./globals.css";

export const metadata = {
  title: "Diabetes Prediction",
  description: "Predict diabetes risk from clinical features",
};

/** HTML shell wrapping all pages. */
export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
