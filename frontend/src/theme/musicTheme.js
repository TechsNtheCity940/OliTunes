import { createTheme } from '@mui/material/styles';

// Define a modern theme with musical inspiration
const musicTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#00c2cb', // Teal accent
      light: '#4ff5ff',
      dark: '#00919a',
      contrastText: '#ffffff',
    },
    secondary: {
      main: '#9c64a6', // Purple accent
      light: '#cf93d7',
      dark: '#6b3877',
      contrastText: '#ffffff',
    },
    background: {
      default: '#121212',
      paper: '#1e1e1e',
    },
    text: {
      primary: '#ffffff',
      secondary: '#b0b0b0',
    },
    success: {
      main: '#4caf50',
      dark: '#388e3c',
    },
    info: {
      main: '#29b6f6',
      dark: '#0288d1',
    },
    warning: {
      main: '#ff9800',
      dark: '#f57c00',
    },
    error: {
      main: '#f44336',
      dark: '#d32f2f',
    },
    // Custom colors for the app
    custom: {
      guitarBody: '#B8651B', // Acoustic guitar body
      guitarNeck: '#614126', // Guitar neck wood
      keyboardBlack: '#121212', // Piano black keys
      keyboardWhite: '#F5F5F5', // Piano white keys
      audioWaveform: '#00c2cb', // Audio waveform color
      tablatureLines: '#555555', // Tablature grid lines
      tablatureNotes: '#FFFFFF', // Tablature notes
      highlight: '#FF5722', // Highlighted elements
      softBackground: 'rgba(255, 255, 255, 0.05)', // Soft overlay for sections
      gradientStart: '#00c2cb', // For gradients
      gradientEnd: '#9c64a6', // For gradients
    },
  },
  typography: {
    fontFamily: "'Poppins', 'Roboto', 'Helvetica', 'Arial', sans-serif",
    h1: {
      fontWeight: 700,
      fontSize: '2.5rem',
      letterSpacing: '-0.01562em',
      background: 'linear-gradient(45deg, #00c2cb 30%, #9c64a6 90%)',
      WebkitBackgroundClip: 'text',
      WebkitTextFillColor: 'transparent',
    },
    h2: {
      fontWeight: 700,
      fontSize: '2rem',
      letterSpacing: '-0.00833em',
    },
    h3: {
      fontWeight: 600,
      fontSize: '1.75rem',
      letterSpacing: '0em',
    },
    h4: {
      fontWeight: 600,
      fontSize: '1.5rem',
      letterSpacing: '0.00735em',
    },
    h5: {
      fontWeight: 500,
      fontSize: '1.25rem',
      letterSpacing: '0em',
    },
    h6: {
      fontWeight: 500,
      fontSize: '1rem',
      letterSpacing: '0.0075em',
    },
    subtitle1: {
      fontWeight: 400,
      fontSize: '1rem',
      letterSpacing: '0.00938em',
    },
    subtitle2: {
      fontWeight: 500,
      fontSize: '0.875rem',
      letterSpacing: '0.00714em',
    },
    body1: {
      fontWeight: 400,
      fontSize: '1rem',
      letterSpacing: '0.00938em',
    },
    body2: {
      fontWeight: 400,
      fontSize: '0.875rem',
      letterSpacing: '0.01071em',
    },
    button: {
      fontWeight: 500,
      fontSize: '0.875rem',
      letterSpacing: '0.02857em',
      textTransform: 'none',
    },
    caption: {
      fontWeight: 400,
      fontSize: '0.75rem',
      letterSpacing: '0.03333em',
    },
    overline: {
      fontWeight: 400,
      fontSize: '0.75rem',
      letterSpacing: '0.08333em',
      textTransform: 'uppercase',
    },
  },
  shape: {
    borderRadius: 12,
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          scrollbarColor: "#6b6b6b #2b2b2b",
          "&::-webkit-scrollbar, & *::-webkit-scrollbar": {
            backgroundColor: "#2b2b2b",
            width: '8px',
            height: '8px',
          },
          "&::-webkit-scrollbar-thumb, & *::-webkit-scrollbar-thumb": {
            borderRadius: 8,
            backgroundColor: "#6b6b6b",
            minHeight: 24,
          },
          "&::-webkit-scrollbar-thumb:focus, & *::-webkit-scrollbar-thumb:focus": {
            backgroundColor: "#959595",
          },
          "&::-webkit-scrollbar-thumb:active, & *::-webkit-scrollbar-thumb:active": {
            backgroundColor: "#959595",
          },
          "&::-webkit-scrollbar-thumb:hover, & *::-webkit-scrollbar-thumb:hover": {
            backgroundColor: "#959595",
          },
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          padding: '8px 16px',
          transition: 'all 0.2s ease-in-out',
          '&:hover': {
            transform: 'translateY(-2px)',
            boxShadow: '0 6px 10px rgba(0, 0, 0, 0.2)',
          },
        },
        contained: {
          boxShadow: '0 2px 5px rgba(0, 0, 0, 0.2)',
        },
        containedPrimary: {
          background: 'linear-gradient(45deg, #00c2cb 30%, #14a9bd 90%)',
        },
        containedSecondary: {
          background: 'linear-gradient(45deg, #9c64a6 30%, #7d4f84 90%)',
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.15)',
          '&.music-paper': {
            position: 'relative',
            '&::before': {
              content: '""',
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              background: 'url(/music-pattern.svg) repeat',
              opacity: 0.03,
              pointerEvents: 'none',
            },
          },
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          transition: 'transform 0.2s, box-shadow 0.2s',
          '&:hover': {
            transform: 'translateY(-4px)',
            boxShadow: '0 12px 20px rgba(0, 0, 0, 0.2)',
          },
        },
      },
    },
    MuiTooltip: {
      styleOverrides: {
        tooltip: {
          borderRadius: 4,
          background: 'rgba(30, 30, 30, 0.9)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          color: '#ffffff',
          boxShadow: '0 4px 8px rgba(0, 0, 0, 0.2)',
          backdropFilter: 'blur(4px)',
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: 16,
          '&.MuiChip-colorPrimary': {
            background: 'linear-gradient(45deg, #00c2cb 30%, #14a9bd 90%)',
          },
          '&.MuiChip-colorSecondary': {
            background: 'linear-gradient(45deg, #9c64a6 30%, #7d4f84 90%)',
          },
        },
      },
    },
    MuiLinearProgress: {
      styleOverrides: {
        root: {
          height: 8,
          borderRadius: 4,
        },
        bar: {
          borderRadius: 4,
        },
        colorPrimary: {
          backgroundColor: 'rgba(0, 194, 203, 0.2)',
        },
        barColorPrimary: {
          background: 'linear-gradient(45deg, #00c2cb 30%, #14a9bd 90%)',
        },
      },
    },
    MuiTabs: {
      styleOverrides: {
        root: {
          borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
        },
        indicator: {
          height: 3,
          borderRadius: '3px 3px 0 0',
          background: 'linear-gradient(45deg, #00c2cb 30%, #9c64a6 90%)',
        },
      },
    },
    MuiTab: {
      styleOverrides: {
        root: {
          transition: 'all 0.2s',
          '&.Mui-selected': {
            color: '#ffffff',
          },
          '&:hover': {
            backgroundColor: 'rgba(255, 255, 255, 0.05)',
          },
        },
      },
    },
    MuiFormLabel: {
      styleOverrides: {
        root: {
          '&.Mui-focused': {
            color: '#00c2cb',
          },
        },
      },
    },
    MuiOutlinedInput: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          '&:hover .MuiOutlinedInput-notchedOutline': {
            borderColor: '#00c2cb',
          },
          '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
            borderColor: '#00c2cb',
            borderWidth: 2,
          },
        },
        notchedOutline: {
          borderColor: 'rgba(255, 255, 255, 0.2)',
        },
      },
    },
    MuiList: {
      styleOverrides: {
        root: {
          padding: 8,
        },
      },
    },
    MuiListItem: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          '&:hover': {
            backgroundColor: 'rgba(255, 255, 255, 0.05)',
          },
          '&.Mui-selected': {
            backgroundColor: 'rgba(0, 194, 203, 0.1)',
            '&:hover': {
              backgroundColor: 'rgba(0, 194, 203, 0.2)',
            },
          },
        },
      },
    },
    MuiSwitch: {
      styleOverrides: {
        root: {
          width: 42,
          height: 26,
          padding: 0,
          margin: 8,
        },
        switchBase: {
          padding: 1,
          '&.Mui-checked': {
            transform: 'translateX(16px)',
            color: '#fff',
            '& + .MuiSwitch-track': {
              backgroundColor: '#00c2cb',
              opacity: 1,
              border: 'none',
            },
          },
        },
        thumb: {
          width: 24,
          height: 24,
        },
        track: {
          borderRadius: 13,
          border: '1px solid rgba(255, 255, 255, 0.2)',
          backgroundColor: 'rgba(255, 255, 255, 0.1)',
          opacity: 1,
          transition: 'background-color 300ms cubic-bezier(0.4, 0, 0.2, 1)',
        },
      },
    },
    MuiTableContainer: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.15)',
        },
      },
    },
    MuiTableHead: {
      styleOverrides: {
        root: {
          '& .MuiTableCell-head': {
            backgroundColor: 'rgba(0, 0, 0, 0.2)',
            color: '#ffffff',
            fontWeight: 600,
          },
        },
      },
    },
    MuiTableRow: {
      styleOverrides: {
        root: {
          '&:nth-of-type(odd)': {
            backgroundColor: 'rgba(255, 255, 255, 0.02)',
          },
          '&:hover': {
            backgroundColor: 'rgba(255, 255, 255, 0.05)',
          },
        },
      },
    },
    MuiTableCell: {
      styleOverrides: {
        root: {
          borderBottom: '1px solid rgba(255, 255, 255, 0.05)',
        },
      },
    },
    MuiMenu: {
      styleOverrides: {
        paper: {
          borderRadius: 12,
          boxShadow: '0 8px 16px rgba(0, 0, 0, 0.3)',
          backgroundColor: 'rgba(30, 30, 30, 0.95)',
          backdropFilter: 'blur(8px)',
        },
      },
    },
    MuiMenuItem: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          margin: '4px 8px',
          '&:hover': {
            backgroundColor: 'rgba(255, 255, 255, 0.05)',
          },
          '&.Mui-selected': {
            backgroundColor: 'rgba(0, 194, 203, 0.1)',
            '&:hover': {
              backgroundColor: 'rgba(0, 194, 203, 0.2)',
            },
          },
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.15)',
          backdropFilter: 'blur(8px)',
          backgroundColor: 'rgba(30, 30, 30, 0.8)',
        },
      },
    },
    MuiDialog: {
      styleOverrides: {
        paper: {
          borderRadius: 16,
          boxShadow: '0 12px 24px rgba(0, 0, 0, 0.3)',
          backdropFilter: 'blur(12px)',
          backgroundImage: 'none',
        },
      },
    },
    MuiPickersDay: {
      styleOverrides: {
        day: {
          '&.Mui-selected': {
            backgroundColor: '#00c2cb',
          },
        },
      },
    },
  },
});

export default musicTheme;
