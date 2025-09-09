  { Routines to provide on-line context-sensitive help}

     PROCEDURE Ds_ShowHelp (Context: Helptype);
       TYPE
         HelpMapType = RECORD
                         men1,men2 : Integer;
                         Head      : String[10];
                       END;

       CONST
         men:Ds_window = (St:(150,80); Fin:(450,380); Sz:(300,300));
         HelpMap : ARRAY[HelpType] OF HelpMapType =
           ( (men1:1; men2:3; Head: 'MAIN'),
             (men1:2; men2:2; Head: 'GEOMETRY'),
             (men1:2; men2:3; Head: 'SNAPSHOT'),
             (men1:2; men2:3; Head: 'DUMP'),
             (men1:2; men2:3; Head: 'HISTORIES'),
             (men1:2; men2:3; Head: 'SETTINGS')     );

         MenuMax = 3;
         Help : ARRAY [1..MenuMax, 1..15] OF String[45] =
             (
                ('<D>isplay graphs - choose "D" or "d" ',
                 '    Display the Geometry or recordings of',
                 '    SNAPSHOTS (contours), DUMPS(vector plots)',
                 '    or HISTORIES (seismograms)',
                 '',
                 '<R>etrieve Images - choose "R" or "r"  ',
                 '    Retrieves and displays images previously',
                 '    created with WVPLOT. These can be viewed',
                 '    as a slide show.',
                 '',
                 '<S>ettings - choose "S" or "s"  ',
                 '    Allows you to change program settings',
                 '    including scales, colour modes, Fourier',
                 '    parameters, multiple graphs windows, etc.',
                 '' ),

                ('<Z>   Zooms the geometric scale',
                 '<z>   Unzooms the geometric scale',
                 '<D/d> Toggles between 2D and 3D views',
                 '<Ctrl-L> Toggles between the default scale for',
                 '      this geometry or links to previous scale',
                 '<'+chr(26)+'>  Unzooms the geometric scale',
                 '',
                 '',
                 '',
                 '',
                 '',
                 '',
                 '',
                 '',
                 '' ),

                ('123456789012345678901234567890123456789012345',
                 ' 2nd line',
                 ' 3rd line',
                 ' 4th line',
                 ' 5th line',
                 ' 6th line',
                 ' 7th line',
                 ' 8th line',
                 ' 9th line',
                 '10th line',
                 '11th line',
                 '12th line',
                 '13th line',
                 '14th line',
                 '123456789012345678901234567890123456789012345')
             );

       VAR
         i,m     : Integer;
         ch      : Char;
         Buffer  : Pointer;
         BufSz   : Word;
       BEGIN {ShowHelp}
         m := HelpMap[Context].men1;

         BufSz := ImageSize (men.st[1],men.st[2],men.fin[2],men.fin[2]);
         IF (MaxAvail < BufSz) THEN Exit;
         GetMem (Buffer, BufSz);
         GetImage (men.st[1],men.st[2],men.fin[2],men.fin[2], Buffer^);

         SetFillStyle (SolidFill,MenuBG);
         Bar(men.st[1],men.st[2],men.fin[1],men.fin[2]);
         SetColor (MenuFG);
         Rectangle (men.st[1]+2,men.st[2]+2,men.fin[1]-2,men.fin[2]-2);
         Rectangle (men.st[1]+4,men.st[2]+4,men.fin[1]-4,men.fin[2]-4);


         REPEAT

           Bar(men.st[1]+5,men.st[2]+5,men.fin[1]-5,men.fin[2]-5);
           SetTextStyle (SmallFont, HorizDir, FontSz-1);
           SetTextJustify (CenterText, CenterText);
           Ds_SetWindow (Men);
           MoveTo ( men.sz[1] DIV 2, 15); OutText ('HELP INFORMATION');

           SetTextJustify (LeftText, CenterText);
           FOR i:=1 TO 15 DO
             BEGIN  MoveTo (15,15*(i+2)); OutText(Help[m,i]);  END;

           SetTextJustify (RightText, CenterText);
           MoveTo (285,270); OutText('<N>ext/<P>revious-page');
           MoveTo (285,285); OutText('<Esc>/Q/q to quit Help');

           ResetViewPort;

           ch:=Readkey;
           CASE ch OF
             'N','n' : inc(m);
             'P','p' : dec(m);
           END;
           IF ((m<1) OR (m>MenuMax)) THEN m:=1;

         UNTIL (ch IN ['Q','q',#27]);

         PutImage (men.st[1],men.st[2], Buffer^, NormalPut);
         FreeMem (Buffer, BufSz);
       END; {Ds_ShowHelp}

